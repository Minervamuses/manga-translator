"""Persistent stage DAG execution with cache, invalidation, resume, and replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.issues import StageName
from ..domain.models import ArtifactRef
from ..storage.job_store import JobStore
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


@dataclass(frozen=True)
class StageOutcome:
    fingerprint: str
    outputs: tuple[ArtifactRef, ...]
    cache_hit: bool


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

    def _raw_response(
        self,
        stage: StageName,
        fingerprint: str,
        request_key: str,
        fetch: Callable[[], bytes],
    ) -> bytes:
        owner_id = f"{self.job_id}:{self.page_id}:{stage.value}:{fingerprint}:{request_key}"
        cached = self.store.find_artifact(owner_type="provider_response", owner_id=owner_id)
        if cached is not None:
            return self.store.artifacts.read_bytes(cached.sha256)
        response = fetch()
        if not isinstance(response, bytes):
            raise TypeError("provider response fetcher must return bytes")
        self.store.store_artifact(
            response,
            media_type="application/octet-stream",
            owner_type="provider_response",
            owner_id=owner_id,
        )
        return response

    def run(
        self,
        *,
        target: StageName = StageName.ENCODE,
        resume: bool = True,
        force_stage: StageName | None = None,
    ) -> dict[StageName, StageOutcome]:
        self.store.interrupt_stale_stage_runs(job_id=self.job_id, page_id=self.page_id)
        forced = downstream_of(force_stage) if force_stage is not None else set()
        if forced:
            self.store.invalidate_stages(
                job_id=self.job_id,
                page_id=self.page_id,
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
                    job_id=self.job_id,
                    page_id=self.page_id,
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
                            f"{fingerprint}:{index}:{payload.role}"
                        ),
                    )
                    for index, payload in enumerate(produced.artifacts)
                )
                self.store.finish_stage(
                    job_id=self.job_id,
                    page_id=self.page_id,
                    stage=name.value,
                    fingerprint=fingerprint,
                    output_hashes=tuple(item.sha256 for item in artifacts),
                )
            except Exception:
                self.store.fail_stage(
                    job_id=self.job_id,
                    page_id=self.page_id,
                    stage=name.value,
                    fingerprint=fingerprint,
                )
                raise
            outcomes[name] = StageOutcome(fingerprint, artifacts, False)
        return outcomes
