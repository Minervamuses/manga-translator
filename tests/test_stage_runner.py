from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from manga_translator.domain.ids import page_id_from_bytes
from manga_translator.domain.issues import StageName
from manga_translator.domain.models import ArtifactRef, PageDocument, SourcePage
from manga_translator.stages.base import (
    ArtifactPayload,
    FingerprintDependencies,
    StageOutputs,
    StageSpec,
)
from manga_translator.stages.fingerprint import stage_fingerprint
from manga_translator.stages.runner import STAGE_DAG, StageRunner
from manga_translator.storage import ArtifactStore, JobStore


def _document() -> PageDocument:
    page_id = page_id_from_bytes(b"source")
    return PageDocument(
        source=SourcePage(
            page_id=page_id,
            original_bytes_sha256=page_id,
            source_path="page.png",
            width=10,
            height=10,
            mode="RGB",
            original_artifact=ArtifactRef(
                sha256=page_id, media_type="image/png", size_bytes=6
            ),
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
    document = _document()
    store.create_job("job")
    store.store_page_document("job", document)
    try:
        yield store, document.source.page_id
    finally:
        store.close()


def _runner(store, page_id, specs, config):
    return StageRunner(
        store=store,
        job_id="job",
        page_id=page_id,
        specs=specs,
        config=config,
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
    store.start_stage(
        job_id="job",
        page_id=page_id,
        stage=StageName.SOURCE.value,
        fingerprint="f" * 64,
        input_hashes=(),
    )
    _runner(store, page_id, _specs(Counter()), {}).run(target=StageName.SOURCE)
    status = store.connection.execute(
        "SELECT status FROM stage_runs WHERE fingerprint=?", ("f" * 64,)
    ).fetchone()[0]
    assert status == "interrupted"


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
