from __future__ import annotations

import multiprocessing
import sqlite3
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import pytest

from manga_translator.domain.ids import page_id_from_bytes
from manga_translator.domain.issues import StageName
from manga_translator.domain.models import ArtifactRef, PageDocument, SourcePage
from manga_translator.stages.base import (
    ArtifactContract,
    ArtifactPayload,
    ArtifactSetContract,
    FingerprintDependencies,
    StageOutputs,
    StageSpec,
)
from manga_translator.stages.fingerprint import select_relevant_config, stage_fingerprint
from manga_translator.stages.runner import STAGE_DAG, StageRunner
from manga_translator.storage import ArtifactStore, JobStore
from manga_translator.storage.job_store import PageRunClaimLostError


def _provider_claim_process_worker(
    database: str,
    artifact_root: str,
    page_id: str,
    barrier,
    fetch_started,
    allow_fetch_to_finish,
    provider_calls,
    results,
) -> None:
    try:
        with JobStore(database, ArtifactStore(artifact_root)) as store:
            runner = StageRunner(
                store=store,
                job_id="job",
                page_id=page_id,
                specs=_specs(Counter()),
                config={},
                provider_response_lease_seconds=0.15,
                provider_response_poll_seconds=0.01,
            )
            barrier.wait(timeout=10)

            def fetch() -> bytes:
                with provider_calls.get_lock():
                    provider_calls.value += 1
                fetch_started.set()
                if not allow_fetch_to_finish.wait(timeout=10):
                    raise TimeoutError("test did not release provider fetch")
                return b'{"translation":"process-safe"}'

            response = runner._raw_response(
                StageName.TRANSLATE,
                "p" * 64,
                "process-request",
                fetch,
            )
            results.put(("ok", response))
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
        results.put(("error", repr(error)))


def _document(original_artifact: ArtifactRef | None = None) -> PageDocument:
    page_id = page_id_from_bytes(b"source")
    return PageDocument(
        source=SourcePage(
            page_id=page_id,
            original_bytes_sha256=page_id,
            source_path="page.png",
            width=10,
            height=10,
            mode="RGB",
            original_artifact=original_artifact
            or ArtifactRef(sha256=page_id, media_type="image/png", size_bytes=6),
        )
    )


def _specs(calls: Counter[StageName]) -> dict[StageName, StageSpec]:
    result = {}
    for name, dependencies in STAGE_DAG.items():
        config_keys = {
            StageName.DETECT: ("detector.threshold",),
            StageName.OCR: ("ocr.language",),
            StageName.TRANSLATE: ("glossary_revision",),
            StageName.LAYOUT: ("font",),
        }.get(name, ())

        def run(_context, inputs, current=name, current_dependencies=dependencies):
            calls[current] += 1
            content = current.value.encode() + b":" + b",".join(
                artifact.sha256.encode()
                for dependency in current_dependencies
                for artifact in inputs.upstream[dependency]
            )
            return StageOutputs((ArtifactPayload(content, "application/test", "primary"),))

        result[name] = StageSpec(
            name=name,
            dependencies=dependencies,
            run=run,
            code_revision="code-v1",
            config_keys=config_keys,
            fingerprint_dependencies=FingerprintDependencies(
                model_hashes=("m" * 64,) if name in {StageName.DETECT, StageName.OCR} else (),
                dependency_versions={"runtime": "1"},
                prompt_revision="prompt-v1" if name is StageName.TRANSLATE else "",
            ),
        )
    return result


@pytest.fixture
def persisted_job(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    store = JobStore(tmp_path / "jobs.sqlite3", artifacts)
    store.create_job("job")
    original_artifact = store.store_artifact(
        b"source", media_type="image/png", owner_type="source", owner_id="page"
    )
    document = _document(original_artifact)
    store.store_page_document("job", document)
    try:
        yield store, document.source.page_id
    finally:
        store.close()


def _runner(store, page_id, specs, config, **kwargs):
    return StageRunner(
        store=store,
        job_id="job",
        page_id=page_id,
        specs=specs,
        config=config,
        **kwargs,
    )


def test_font_change_hits_detect_ocr_translation_but_reruns_layout_downstream(
    persisted_job,
) -> None:
    store, page_id = persisted_job
    calls: Counter[StageName] = Counter()
    specs = _specs(calls)
    config = {"font": "font-a", "glossary_revision": "g1"}
    _runner(store, page_id, specs, config).run()
    first_counts = calls.copy()
    outcomes = _runner(store, page_id, specs, {**config, "font": "font-b"}).run()

    assert calls[StageName.DETECT] == first_counts[StageName.DETECT]
    assert calls[StageName.OCR] == first_counts[StageName.OCR]
    assert calls[StageName.TRANSLATE] == first_counts[StageName.TRANSLATE]
    assert calls[StageName.LAYOUT] == first_counts[StageName.LAYOUT] + 1
    assert calls[StageName.INPAINT_RENDER] == first_counts[StageName.INPAINT_RENDER] + 1
    assert calls[StageName.ENCODE] == first_counts[StageName.ENCODE] + 1
    assert outcomes[StageName.DETECT].cache_hit


def test_glossary_change_reruns_translation_and_downstream(persisted_job) -> None:
    store, page_id = persisted_job
    calls: Counter[StageName] = Counter()
    specs = _specs(calls)
    config = {"font": "font-a", "glossary_revision": "g1"}
    _runner(store, page_id, specs, config).run()
    before = calls.copy()
    _runner(store, page_id, specs, {**config, "glossary_revision": "g2"}).run()

    assert calls[StageName.DETECT] == before[StageName.DETECT]
    assert calls[StageName.OCR] == before[StageName.OCR]
    assert calls[StageName.TRANSLATE] == before[StageName.TRANSLATE] + 1
    assert calls[StageName.LAYOUT] == before[StageName.LAYOUT] + 1
    assert calls[StageName.ENCODE] == before[StageName.ENCODE] + 1


def test_force_stage_invalidates_it_and_every_downstream(persisted_job) -> None:
    store, page_id = persisted_job
    calls: Counter[StageName] = Counter()
    specs = _specs(calls)
    runner = _runner(store, page_id, specs, {})
    runner.run()
    before = calls.copy()
    runner.run(force_stage=StageName.OCR)

    assert calls[StageName.SOURCE] == before[StageName.SOURCE]
    assert calls[StageName.DETECT] == before[StageName.DETECT]
    assert calls[StageName.OCR] == before[StageName.OCR] + 1
    assert calls[StageName.TRANSLATE] == before[StageName.TRANSLATE] + 1
    assert calls[StageName.LAYOUT] == before[StageName.LAYOUT] + 1
    assert calls[StageName.INPAINT_RENDER] == before[StageName.INPAINT_RENDER] + 1
    assert calls[StageName.ENCODE] == before[StageName.ENCODE] + 1


def test_stale_running_stage_becomes_interrupted(persisted_job) -> None:
    store, page_id = persisted_job
    stale_claim = store.acquire_page_run_claim(
        job_id="job", page_id=page_id, lease_seconds=60
    )
    assert stale_claim is not None
    store.start_stage(
        job_id="job",
        page_id=page_id,
        stage=StageName.SOURCE.value,
        fingerprint="f" * 64,
        input_hashes=(),
        claim=stale_claim,
    )
    store.connection.execute(
        """
        UPDATE page_run_claims SET lease_expires_at_ms=1
        WHERE job_id=? AND page_id=?
        """,
        ("job", page_id),
    )
    _runner(
        store,
        page_id,
        _specs(Counter()),
        {},
        page_run_lease_seconds=0.12,
        page_run_poll_seconds=0.01,
    ).run(target=StageName.SOURCE)
    status = store.connection.execute(
        "SELECT status FROM stage_runs WHERE fingerprint=?", ("f" * 64,)
    ).fetchone()[0]
    assert status == "interrupted"
    assert not store.release_page_run_claim(stale_claim)


def test_full_runners_serialize_one_page_and_second_runner_uses_cache(
    persisted_job, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, page_id = persisted_job
    first_calls: Counter[StageName] = Counter()
    second_calls: Counter[StageName] = Counter()
    first_specs = _specs(first_calls)
    second_specs = _specs(second_calls)
    source_started = Event()
    allow_source_to_finish = Event()
    first_finished = Event()
    second_finished = Event()
    initial_renewal_finished = Event()
    failures: list[Exception] = []
    results: dict[str, dict[StageName, object]] = {}

    original_renew = JobStore.renew_page_run_claim

    def tracked_renew(self, claim, *, lease_seconds):
        renewed = original_renew(self, claim, lease_seconds=lease_seconds)
        initial_renewal_finished.set()
        return renewed

    monkeypatch.setattr(JobStore, "renew_page_run_claim", tracked_renew)
    original_source = first_specs[StageName.SOURCE].run

    def blocked_source(context, inputs):
        assert initial_renewal_finished.is_set()
        source_started.set()
        if not allow_source_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release the source stage")
        return original_source(context, inputs)

    first_specs[StageName.SOURCE] = replace(
        first_specs[StageName.SOURCE], run=blocked_source
    )

    def worker(label: str, specs, finished: Event) -> None:
        try:
            with JobStore(store.database, ArtifactStore(store.artifacts.root)) as isolated:
                results[label] = _runner(
                    isolated,
                    page_id,
                    specs,
                    {},
                    page_run_lease_seconds=0.12,
                    page_run_poll_seconds=0.01,
                ).run(target=StageName.SOURCE)
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
            failures.append(error)
        finally:
            finished.set()

    first_thread = Thread(target=worker, args=("first", first_specs, first_finished))
    first_thread.start()
    assert source_started.wait(timeout=5)

    second_thread = Thread(target=worker, args=("second", second_specs, second_finished))
    second_thread.start()
    assert not second_finished.wait(timeout=0.35)
    running = store.connection.execute(
        """
        SELECT stage_runs.status, stage_runs.run_token, page_run_claims.claim_token
        FROM stage_runs JOIN page_run_claims USING(job_id, page_id)
        WHERE stage_runs.job_id='job' AND stage_runs.page_id=?
          AND stage_runs.stage='source'
        """,
        (page_id,),
    ).fetchone()
    assert tuple(running) == ("running", running[1], running[1])
    assert second_calls[StageName.SOURCE] == 0

    allow_source_to_finish.set()
    assert first_finished.wait(timeout=5)
    assert second_finished.wait(timeout=5)
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not failures
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not results["first"][StageName.SOURCE].cache_hit
    assert results["second"][StageName.SOURCE].cache_hit
    assert first_calls[StageName.SOURCE] == 1
    assert second_calls[StageName.SOURCE] == 0
    assert store.connection.execute("SELECT count(*) FROM page_run_claims").fetchone()[0] == 0


def test_expired_page_runner_is_fenced_from_replacement_stage_and_artifacts(
    persisted_job,
) -> None:
    old_store, page_id = persisted_job
    fingerprint = "f" * 64
    old_claim = old_store.acquire_page_run_claim(
        job_id="job", page_id=page_id, lease_seconds=60
    )
    assert old_claim is not None
    old_store.start_stage(
        job_id="job",
        page_id=page_id,
        stage=StageName.SOURCE.value,
        fingerprint=fingerprint,
        input_hashes=(),
        claim=old_claim,
    )
    old_owner = f"job:{page_id}:source:{fingerprint}:{old_claim.claim_token}:0:primary"
    old_store.store_artifact(
        b"old-output",
        media_type="application/test",
        owner_type="stage_output",
        owner_id=old_owner,
    )
    old_store.connection.execute(
        """
        UPDATE page_run_claims SET lease_expires_at_ms=1
        WHERE job_id=? AND page_id=?
        """,
        ("job", page_id),
    )

    with JobStore(old_store.database, ArtifactStore(old_store.artifacts.root)) as replacement:
        new_claim = replacement.acquire_page_run_claim(
            job_id="job", page_id=page_id, lease_seconds=60
        )
        assert new_claim is not None
        assert replacement.interrupt_stale_stage_runs(claim=new_claim) == 1
        replacement.start_stage(
            job_id="job",
            page_id=page_id,
            stage=StageName.SOURCE.value,
            fingerprint=fingerprint,
            input_hashes=(),
            claim=new_claim,
        )
        new_owner = f"job:{page_id}:source:{fingerprint}:{new_claim.claim_token}:0:primary"
        winner = replacement.store_artifact(
            b"winner",
            media_type="application/test",
            owner_type="stage_output",
            owner_id=new_owner,
        )
        late_owner = f"job:{page_id}:source:{fingerprint}:{old_claim.claim_token}:1:late"
        late = old_store.store_artifact(
            b"late-old-output",
            media_type="application/test",
            owner_type="stage_output",
            owner_id=late_owner,
        )

        with pytest.raises(PageRunClaimLostError, match="page run claim was lost"):
            old_store.finish_stage(
                job_id="job",
                page_id=page_id,
                stage=StageName.SOURCE.value,
                fingerprint=fingerprint,
                output_hashes=(late.sha256,),
                claim=old_claim,
            )
        assert not old_store.fail_stage(
            job_id="job",
            page_id=page_id,
            stage=StageName.SOURCE.value,
            fingerprint=fingerprint,
            claim=old_claim,
        )
        with pytest.raises(PageRunClaimLostError, match="page run claim was lost"):
            old_store.invalidate_stages(
                claim=old_claim, stages={StageName.SOURCE.value}
            )
        assert (
            replacement.connection.execute(
                """
                SELECT status FROM stage_runs
                WHERE job_id='job' AND page_id=? AND stage='source' AND fingerprint=?
                """,
                (page_id, fingerprint),
            ).fetchone()[0]
            == "running"
        )
        assert (
            old_store.discard_stage_attempt_outputs(
                job_id="job",
                page_id=page_id,
                stage=StageName.SOURCE.value,
                fingerprint=fingerprint,
                claim=old_claim,
            )
            == 1
        )
        replacement.finish_stage(
            job_id="job",
            page_id=page_id,
            stage=StageName.SOURCE.value,
            fingerprint=fingerprint,
            output_hashes=(winner.sha256,),
            claim=new_claim,
        )
        row = replacement.connection.execute(
            """
            SELECT status, output_hashes_json, run_token FROM stage_runs
            WHERE job_id='job' AND page_id=? AND stage='source' AND fingerprint=?
            """,
            (page_id, fingerprint),
        ).fetchone()
        assert tuple(row) == ("succeeded", f'["{winner.sha256}"]', new_claim.claim_token)
        assert replacement.find_artifact(owner_type="stage_output", owner_id=old_owner) is None
        assert replacement.find_artifact(owner_type="stage_output", owner_id=late_owner) is None
        assert replacement.find_artifact(owner_type="stage_output", owner_id=new_owner) == winner
        assert replacement.release_page_run_claim(new_claim)

    assert not old_store.release_page_run_claim(old_claim)


def test_raw_provider_response_is_replayed_after_interrupted_stage(persisted_job) -> None:
    store, page_id = persisted_job
    calls: Counter[StageName] = Counter()
    specs = _specs(calls)
    provider_calls = 0

    def fetch() -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return b'{"translation":"saved"}'

    def interrupted(context, _inputs):
        context.get_or_fetch_raw_response("request-1", fetch)
        raise RuntimeError("simulated kill after provider response")

    specs[StageName.TRANSLATE] = replace(specs[StageName.TRANSLATE], run=interrupted)
    with pytest.raises(RuntimeError, match="simulated kill"):
        _runner(store, page_id, specs, {}).run()

    def recovered(context, _inputs):
        response = context.get_or_fetch_raw_response("request-1", fetch)
        return StageOutputs((ArtifactPayload(response, "application/json", "translation"),))

    specs[StageName.TRANSLATE] = replace(specs[StageName.TRANSLATE], run=recovered)
    outcomes = _runner(store, page_id, specs, {}).run()

    assert provider_calls == 1
    assert outcomes[StageName.DETECT].cache_hit
    assert outcomes[StageName.OCR].cache_hit
    assert store.artifacts.read_bytes(outcomes[StageName.TRANSLATE].outputs[0].sha256).startswith(
        b'{"translation"'
    )


def test_two_runners_share_one_provider_fetch_while_lease_is_renewed(
    persisted_job,
) -> None:
    store, page_id = persisted_job
    barrier = Barrier(3)
    fetch_started = Event()
    allow_fetch_to_finish = Event()
    observation_delay = Event()
    calls_lock = Lock()
    provider_calls = 0
    responses: list[bytes] = []
    failures: list[Exception] = []

    def fetch() -> bytes:
        nonlocal provider_calls
        with calls_lock:
            provider_calls += 1
        fetch_started.set()
        if not allow_fetch_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release provider fetch")
        return b'{"translation":"single-fetch"}'

    def worker() -> None:
        try:
            with JobStore(store.database, ArtifactStore(store.artifacts.root)) as isolated:
                runner = _runner(
                    isolated,
                    page_id,
                    _specs(Counter()),
                    {},
                    provider_response_lease_seconds=0.12,
                    provider_response_poll_seconds=0.01,
                )
                barrier.wait(timeout=5)
                responses.append(
                    runner._raw_response(
                        StageName.TRANSLATE,
                        "f" * 64,
                        "same-request",
                        fetch,
                    )
                )
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
            failures.append(error)

    threads = [Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    fetch_observed = fetch_started.wait(timeout=5)
    observation_delay.wait(timeout=0.4)
    with calls_lock:
        observed_calls = provider_calls
    allow_fetch_to_finish.set()
    for thread in threads:
        thread.join(timeout=5)

    assert fetch_observed
    assert observed_calls == 1
    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    assert responses == [b'{"translation":"single-fetch"}'] * 2
    assert (
        store.connection.execute(
            "SELECT count(*) FROM provider_response_claims"
        ).fetchone()[0]
        == 0
    )
    assert (
        store.connection.execute(
            """
            SELECT count(*) FROM artifact_references
            WHERE owner_type='provider_response'
            """
        ).fetchone()[0]
        == 1
    )


def test_provider_fetch_claim_is_shared_across_processes(persisted_job) -> None:
    store, page_id = persisted_job
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    fetch_started = context.Event()
    allow_fetch_to_finish = context.Event()
    provider_calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_provider_claim_process_worker,
            args=(
                str(store.database),
                str(store.artifacts.root),
                page_id,
                barrier,
                fetch_started,
                allow_fetch_to_finish,
                provider_calls,
                results,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    fetch_observed = fetch_started.wait(timeout=10)
    Event().wait(timeout=0.5)
    with provider_calls.get_lock():
        observed_calls = provider_calls.value
    allow_fetch_to_finish.set()
    for process in processes:
        process.join(timeout=10)

    assert fetch_observed
    assert observed_calls == 1
    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=2) for _ in processes) == [
        ("ok", b'{"translation":"process-safe"}'),
        ("ok", b'{"translation":"process-safe"}'),
    ]


def test_failed_and_expired_provider_leases_are_recoverable(persisted_job) -> None:
    store, page_id = persisted_job
    runner = _runner(
        store,
        page_id,
        _specs(Counter()),
        {},
        provider_response_lease_seconds=10,
        provider_response_poll_seconds=0.01,
    )
    fingerprint = "e" * 64
    expired_owner = (
        f"job:{page_id}:{StageName.TRANSLATE.value}:{fingerprint}:expired-request"
    )
    expired = store.acquire_provider_response_claim(
        owner_id=expired_owner, lease_seconds=60
    )
    assert expired is not None
    store.connection.execute(
        "UPDATE provider_response_claims SET lease_expires_at_ms=1 WHERE owner_id=?",
        (expired_owner,),
    )

    assert (
        runner._raw_response(
            StageName.TRANSLATE,
            fingerprint,
            "expired-request",
            lambda: b"recovered-expired",
        )
        == b"recovered-expired"
    )
    assert not store.release_provider_response_claim(expired)

    def fail_fetch() -> bytes:
        raise RuntimeError("simulated provider failure")

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        runner._raw_response(
            StageName.TRANSLATE,
            fingerprint,
            "failed-request",
            fail_fetch,
        )
    assert (
        store.connection.execute(
            "SELECT count(*) FROM provider_response_claims"
        ).fetchone()[0]
        == 0
    )
    assert (
        runner._raw_response(
            StageName.TRANSLATE,
            fingerprint,
            "failed-request",
            lambda: b"recovered-failure",
        )
        == b"recovered-failure"
    )


def test_missing_cached_artifact_invalidates_and_recomputes(persisted_job) -> None:
    store, page_id = persisted_job
    calls: Counter[StageName] = Counter()
    specs = _specs(calls)
    first = _runner(store, page_id, specs, {}).run(target=StageName.DETECT)
    store.artifacts.delete(first[StageName.DETECT].outputs[0].sha256)
    before = calls[StageName.DETECT]
    second = _runner(store, page_id, specs, {}).run(target=StageName.DETECT)
    assert calls[StageName.DETECT] == before + 1
    assert not second[StageName.DETECT].cache_hit


def test_every_declared_dependency_participates_in_fingerprint() -> None:
    spec = _specs(Counter())[StageName.TRANSLATE]
    baseline = stage_fingerprint(spec, upstream_output_hashes=("a" * 64,), config={})
    changed = replace(
        spec,
        fingerprint_dependencies=replace(
            spec.fingerprint_dependencies,
            glossary_revision="glossary-v2",
            entity_revision="entities-v2",
            model_hashes=("b" * 64,),
            font_hashes=("c" * 64,),
            preprocess_revision="preprocess-v2",
            prompt_revision="prompt-v2",
            schema_revision="schema-v2",
            dependency_versions={"runtime": "2"},
        ),
    )
    assert stage_fingerprint(changed, upstream_output_hashes=("a" * 64,), config={}) != baseline


def test_missing_config_path_cannot_collide_with_real_mapping_value() -> None:
    missing = select_relevant_config({}, ("provider.capability",))
    present = select_relevant_config(
        {"provider": {"capability": {"missing": True}}},
        ("provider.capability",),
    )

    assert missing == {"provider.capability": {"present": False, "value": None}}
    assert present == {
        "provider.capability": {"present": True, "value": {"missing": True}}
    }
    assert missing != present


def test_stage_output_must_match_typed_artifact_contract(persisted_job) -> None:
    store, page_id = persisted_job
    specs = _specs(Counter())
    specs[StageName.SOURCE] = replace(
        specs[StageName.SOURCE],
        output_contract=ArtifactSetContract(
            required=(ArtifactContract("source", "application/octet-stream"),)
        ),
    )

    with pytest.raises(ValueError, match="output contract mismatch"):
        _runner(store, page_id, specs, {}).run(target=StageName.SOURCE)

    status = store.connection.execute(
        "SELECT status FROM stage_runs WHERE job_id='job' AND page_id=? AND stage='source'",
        (page_id,),
    ).fetchone()[0]
    assert status == "failed"
