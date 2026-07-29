"""Persistent stage DAG execution with cache, invalidation, resume, and replay."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread
from time import sleep
from typing import Any

from ..domain.issues import StageName
from ..domain.models import ArtifactRef, PageDocument
from ..storage.job_store import (
    JobStore,
    PageRunClaim,
    PageRunClaimLostError,
    ProviderResponseClaim,
    ProviderResponseClaimLostError,
)
from .base import StageContext, StageInputs, StageOutputs, StageSpec
from .fingerprint import stage_fingerprint

STAGE_DAG: dict[StageName, tuple[StageName, ...]] = {
    StageName.SOURCE: (),
    StageName.DETECT: (StageName.SOURCE,),
    StageName.STYLE: (StageName.SOURCE, StageName.DETECT),
    StageName.SAFE_REGION: (StageName.SOURCE, StageName.DETECT),
    StageName.OCR: (StageName.SOURCE, StageName.DETECT),
    StageName.ORDER: (StageName.SOURCE, StageName.DETECT),
    StageName.TRANSLATE: (StageName.OCR, StageName.ORDER),
    StageName.LAYOUT: (StageName.TRANSLATE, StageName.STYLE, StageName.SAFE_REGION),
    StageName.INPAINT_RENDER: (
        StageName.SOURCE,
        StageName.DETECT,
        StageName.TRANSLATE,
        StageName.LAYOUT,
    ),
    StageName.ENCODE: (StageName.INPAINT_RENDER,),
}

TOPOLOGICAL_ORDER = tuple(STAGE_DAG)
DEFAULT_PROVIDER_RESPONSE_LEASE_SECONDS = 30.0
DEFAULT_PROVIDER_RESPONSE_POLL_SECONDS = 0.05
DEFAULT_PAGE_RUN_LEASE_SECONDS = 30.0
DEFAULT_PAGE_RUN_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class StageOutcome:
    fingerprint: str
    outputs: tuple[ArtifactRef, ...]
    cache_hit: bool


@dataclass(frozen=True)
class StageFailureContext:
    stage: StageName
    fingerprint: str
    partial_outcomes: Mapping[StageName, StageOutcome]


StageCheckpoint = Callable[
    [StageName, Mapping[StageName, StageOutcome]], PageDocument
]


def downstream_of(stage: StageName) -> set[StageName]:
    result = {stage}
    changed = True
    while changed:
        changed = False
        for candidate, dependencies in STAGE_DAG.items():
            if candidate not in result and any(item in result for item in dependencies):
                result.add(candidate)
                changed = True
    return result


def ancestors_of(stage: StageName) -> set[StageName]:
    result = {stage}
    pending = [stage]
    while pending:
        current = pending.pop()
        for dependency in STAGE_DAG[current]:
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return result


class StageRunner:
    def __init__(
        self,
        *,
        store: JobStore,
        job_id: str,
        page_id: str,
        specs: Mapping[StageName, StageSpec],
        config: Mapping[str, Any],
        provider_response_lease_seconds: float = DEFAULT_PROVIDER_RESPONSE_LEASE_SECONDS,
        provider_response_poll_seconds: float = DEFAULT_PROVIDER_RESPONSE_POLL_SECONDS,
        page_run_lease_seconds: float = DEFAULT_PAGE_RUN_LEASE_SECONDS,
        page_run_poll_seconds: float = DEFAULT_PAGE_RUN_POLL_SECONDS,
        checkpoint: StageCheckpoint | None = None,
    ) -> None:
        missing = set(STAGE_DAG) - set(specs)
        if missing:
            raise ValueError(f"missing stage specs: {sorted(item.value for item in missing)}")
        for name, expected in STAGE_DAG.items():
            if specs[name].name != name or specs[name].dependencies != expected:
                raise ValueError(f"stage {name.value} does not match the fixed DAG")
        self.store = store
        self.job_id = job_id
        self.page_id = page_id
        self.specs = dict(specs)
        self.config = config
        if provider_response_lease_seconds <= 0:
            raise ValueError("provider_response_lease_seconds must be positive")
        if provider_response_poll_seconds <= 0:
            raise ValueError("provider_response_poll_seconds must be positive")
        if page_run_lease_seconds <= 0:
            raise ValueError("page_run_lease_seconds must be positive")
        if page_run_poll_seconds <= 0:
            raise ValueError("page_run_poll_seconds must be positive")
        self.provider_response_lease_seconds = provider_response_lease_seconds
        self.provider_response_poll_seconds = provider_response_poll_seconds
        self.page_run_lease_seconds = page_run_lease_seconds
        self.page_run_poll_seconds = page_run_poll_seconds
        self.checkpoint = checkpoint

    def _acquire_page_run_claim(self) -> PageRunClaim:
        while True:
            claim = self.store.acquire_page_run_claim(
                job_id=self.job_id,
                page_id=self.page_id,
                lease_seconds=self.page_run_lease_seconds,
            )
            if claim is not None:
                return claim
            sleep(self.page_run_poll_seconds)

    @contextmanager
    def _maintain_page_run_claim(self, claim: PageRunClaim) -> Iterator[None]:
        stopped = Event()
        ready = Event()
        failures: list[Exception] = []
        interval = min(5.0, self.page_run_lease_seconds / 3)

        def heartbeat() -> None:
            try:
                with JobStore(self.store.database, self.store.artifacts) as lease_store:
                    lease_store.renew_page_run_claim(
                        claim, lease_seconds=self.page_run_lease_seconds
                    )
                    ready.set()
                    while not stopped.wait(interval):
                        lease_store.renew_page_run_claim(
                            claim, lease_seconds=self.page_run_lease_seconds
                        )
            except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
                failures.append(error)
                ready.set()

        thread = Thread(
            target=heartbeat,
            name=f"page-run-lease-{claim.claim_token[:8]}",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            stopped.set()
            thread.join(timeout=5)
            raise PageRunClaimLostError(
                f"page run lease heartbeat did not start for {claim.job_id}:{claim.page_id}"
            )
        if failures:
            stopped.set()
            thread.join(timeout=5)
            raise PageRunClaimLostError(
                f"page run lease heartbeat failed for {claim.job_id}:{claim.page_id}"
            ) from failures[0]

        try:
            yield
        except BaseException:
            stopped.set()
            thread.join(timeout=5)
            raise
        else:
            stopped.set()
            thread.join(timeout=5)
            if thread.is_alive():
                raise PageRunClaimLostError(
                    f"page run lease heartbeat did not stop for {claim.job_id}:{claim.page_id}"
                )
            if failures:
                raise PageRunClaimLostError(
                    f"page run lease was lost for {claim.job_id}:{claim.page_id}"
                ) from failures[0]
            self.store.renew_page_run_claim(
                claim, lease_seconds=self.page_run_lease_seconds
            )

    def _cached_raw_response(self, owner_id: str) -> bytes | None:
        cached = self.store.find_artifact(
            owner_type="provider_response", owner_id=owner_id
        )
        if cached is None:
            return None
        return self.store.artifacts.read_bytes(
            cached.sha256, expected_size=cached.size_bytes
        )

    @contextmanager
    def _maintain_provider_response_claim(
        self, claim: ProviderResponseClaim
    ) -> Iterator[None]:
        stopped = Event()
        ready = Event()
        failures: list[Exception] = []
        interval = min(5.0, self.provider_response_lease_seconds / 3)

        def heartbeat() -> None:
            try:
                with JobStore(self.store.database, self.store.artifacts) as lease_store:
                    lease_store.renew_provider_response_claim(
                        claim,
                        lease_seconds=self.provider_response_lease_seconds,
                    )
                    ready.set()
                    while not stopped.wait(interval):
                        lease_store.renew_provider_response_claim(
                            claim,
                            lease_seconds=self.provider_response_lease_seconds,
                        )
            except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
                failures.append(error)
                ready.set()

        thread = Thread(
            target=heartbeat,
            name=f"provider-response-lease-{claim.claim_token[:8]}",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            stopped.set()
            thread.join(timeout=5)
            raise ProviderResponseClaimLostError(
                f"provider response lease heartbeat did not start for {claim.owner_id}"
            )
        if failures:
            stopped.set()
            thread.join(timeout=5)
            raise ProviderResponseClaimLostError(
                f"provider response lease heartbeat failed for {claim.owner_id}"
            ) from failures[0]

        try:
            yield
        except BaseException:
            stopped.set()
            thread.join(timeout=5)
            raise
        else:
            stopped.set()
            thread.join(timeout=5)
            if thread.is_alive():
                raise ProviderResponseClaimLostError(
                    f"provider response lease heartbeat did not stop for {claim.owner_id}"
                )
            if failures:
                raise ProviderResponseClaimLostError(
                    f"provider response lease was lost for {claim.owner_id}"
                ) from failures[0]
            self.store.renew_provider_response_claim(
                claim,
                lease_seconds=self.provider_response_lease_seconds,
            )

    def _raw_response(
        self,
        stage: StageName,
        fingerprint: str,
        request_key: str,
        fetch: Callable[[], bytes],
    ) -> bytes:
        owner_id = f"{self.job_id}:{self.page_id}:{stage.value}:{fingerprint}:{request_key}"
        while True:
            cached = self._cached_raw_response(owner_id)
            if cached is not None:
                return cached
            claim = self.store.acquire_provider_response_claim(
                owner_id=owner_id,
                lease_seconds=self.provider_response_lease_seconds,
            )
            if claim is None:
                sleep(self.provider_response_poll_seconds)
                continue
            try:
                cached = self._cached_raw_response(owner_id)
                if cached is not None:
                    self.store.release_provider_response_claim(claim)
                    return cached
                with self._maintain_provider_response_claim(claim):
                    response = fetch()
                    if not isinstance(response, bytes):
                        raise TypeError("provider response fetcher must return bytes")
                self.store.complete_provider_response_claim(
                    claim,
                    response,
                    media_type="application/octet-stream",
                )
                return response
            except BaseException:
                self.store.release_provider_response_claim(claim)
                raise

    def run(
        self,
        *,
        target: StageName = StageName.ENCODE,
        resume: bool = True,
        force_stage: StageName | None = None,
    ) -> dict[StageName, StageOutcome]:
        claim = self._acquire_page_run_claim()
        try:
            with self._maintain_page_run_claim(claim):
                return self._run_claimed(
                    claim=claim,
                    target=target,
                    resume=resume,
                    force_stage=force_stage,
                )
        finally:
            self.store.release_page_run_claim(claim)

    def _run_claimed(
        self,
        *,
        claim: PageRunClaim,
        target: StageName,
        resume: bool,
        force_stage: StageName | None,
    ) -> dict[StageName, StageOutcome]:
        self.store.interrupt_stale_stage_runs(claim=claim)
        forced = downstream_of(force_stage) if force_stage is not None else set()
        if forced:
            self.store.invalidate_stages(
                claim=claim,
                stages={item.value for item in forced},
            )
        required = ancestors_of(target)
        outcomes: dict[StageName, StageOutcome] = {}
        for name in TOPOLOGICAL_ORDER:
            if name not in required:
                continue
            spec = self.specs[name]
            upstream = {dependency: outcomes[dependency].outputs for dependency in spec.dependencies}
            for dependency, contract in spec.input_contracts.items():
                contract.validate_refs(upstream[dependency], dependency=dependency)
            upstream_hashes = tuple(
                dependency_hash
                for dependency in spec.dependencies
                for dependency_hash in (
                    outcomes[dependency].fingerprint,
                    *(artifact.sha256 for artifact in upstream[dependency]),
                )
            )
            fingerprint = stage_fingerprint(
                spec, upstream_output_hashes=upstream_hashes, config=self.config
            )
            cached = None
            if resume and name not in forced:
                cached = self.store.cached_stage_outputs(
                    claim=claim,
                    stage=name.value,
                    fingerprint=fingerprint,
                )
            if cached is not None:
                outcomes[name] = StageOutcome(fingerprint, cached, True)
                continue
            self.store.start_stage(
                job_id=self.job_id,
                page_id=self.page_id,
                stage=name.value,
                fingerprint=fingerprint,
                input_hashes=upstream_hashes,
                claim=claim,
            )
            context = StageContext(
                job_id=self.job_id,
                page_id=self.page_id,
                stage=name,
                fingerprint=fingerprint,
                config=self.config,
                raw_response=lambda key, fetch, current=name, current_fp=fingerprint: (
                    self._raw_response(current, current_fp, key, fetch)
                ),
            )
            try:
                produced = spec.run(context, StageInputs(upstream))
                if not isinstance(produced, StageOutputs):
                    raise TypeError("stage function must return StageOutputs")
                if spec.output_contract is not None:
                    spec.output_contract.validate_payloads(produced.artifacts)
                for payload in produced.artifacts:
                    if not isinstance(payload.data, bytes):
                        raise TypeError("stage artifact payload data must be bytes")
                    if not payload.media_type or not payload.role:
                        raise ValueError("stage artifact media_type and role must not be empty")
                artifacts = tuple(
                    self.store.store_artifact(
                        payload.data,
                        media_type=payload.media_type,
                        owner_type="stage_output",
                        owner_id=(
                            f"{self.job_id}:{self.page_id}:{name.value}:"
                            f"{fingerprint}:{claim.claim_token}:{index}:{payload.role}"
                        ),
                    )
                    for index, payload in enumerate(produced.artifacts)
                )
                outcome = StageOutcome(fingerprint, artifacts, False)
                checkpoint_document = (
                    self.checkpoint(name, {**outcomes, name: outcome})
                    if self.checkpoint is not None
                    else None
                )
                self.store.finish_stage(
                    job_id=self.job_id,
                    page_id=self.page_id,
                    stage=name.value,
                    fingerprint=fingerprint,
                    output_hashes=tuple(item.sha256 for item in artifacts),
                    claim=claim,
                    document=checkpoint_document,
                )
            except Exception as error:
                self.store.fail_stage(
                    job_id=self.job_id,
                    page_id=self.page_id,
                    stage=name.value,
                    fingerprint=fingerprint,
                    claim=claim,
                )
                self.store.discard_stage_attempt_outputs(
                    job_id=self.job_id,
                    page_id=self.page_id,
                    stage=name.value,
                    fingerprint=fingerprint,
                    claim=claim,
                )
                error.stage_failure_context = StageFailureContext(  # type: ignore[attr-defined]
                    stage=name,
                    fingerprint=fingerprint,
                    partial_outcomes=dict(outcomes),
                )
                raise
            outcomes[name] = outcome
        return outcomes
