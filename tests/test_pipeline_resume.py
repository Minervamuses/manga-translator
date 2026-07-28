from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from manga_translator import cli as cli_module
from manga_translator import pipeline as pipeline_module
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.detector import TextGroup, TextRegion
from manga_translator.domain.issues import StageName
from manga_translator.domain.serialization import canonical_document_bytes
from manga_translator.result import PageResult
from manga_translator.storage import ArtifactStore, JobStore


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        openrouter=OpenRouterConfig(api_key="secret-not-persisted", model="test/model"),
        paths=PathsConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            glossary=tmp_path / "glossary.json",
            font=tmp_path / "font.ttf",
            font_fallback=tmp_path / "fallback.ttf",
        ),
    )


def _source(config: AppConfig) -> Path:
    path = config.paths.input_dir / "page.png"
    path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(path), np.full((40, 60, 3), 255, dtype=np.uint8))
    return path


def _legacy_result(path: Path) -> PageResult:
    page_id = hashlib.sha256(path.read_bytes()).hexdigest()
    regions = [
        TextRegion(
            id="legacy-r1",
            x=4,
            y=5,
            w=12,
            h=18,
            confidence=0.9,
            local_mask=np.full((18, 12), 255, dtype=np.uint8),
        ),
        TextRegion(
            id="legacy-r2",
            x=30,
            y=5,
            w=12,
            h=18,
            confidence=0.8,
            local_mask=np.full((18, 12), 255, dtype=np.uint8),
        ),
    ]
    groups = [
        TextGroup(
            id="g1",
            region_ids=["legacy-r1"],
            bbox=(4, 5, 12, 18),
            vertical=True,
            ocr_text="猫",
            ocr_text_norm="猫",
            ocr_confidence=0.95,
            ocr_source="ensemble",
            translation="貓",
            translation_valid=True,
            status="ready",
            mapping_region_key="request:g1",
        ),
        TextGroup(
            id="g2",
            region_ids=["legacy-r2"],
            bbox=(30, 5, 12, 18),
            vertical=True,
            ocr_text="?",
            ocr_confidence=0.1,
            status="ocr_rejected",
            skip_reason="low_confidence",
        ),
    ]
    return PageResult(
        page_id=page_id,
        source_path=path,
        status="succeeded",
        image=np.full((40, 60, 3), 127, dtype=np.uint8),
        regions=regions,
        groups=groups,
        ocr_results=[group.ocr_text for group in groups],
        translations=[group.translation for group in groups],
    )


def _open_document(config: AppConfig, state: Path, page_id: str):
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        return store.load_page_document(job_id="job-1", page_id=page_id)


def test_staged_pipeline_preserves_feature_parity_and_resume_skips_legacy_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls = 0

    def fake_process(path, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _legacy_result(path)

    monkeypatch.setattr(pipeline_module, "process_single_page", fake_process)
    first = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True, dump_json=True
    )
    page_id = first.pages[0].page_id
    first_document = _open_document(config, state, page_id)
    assert first_document is not None
    first_bytes = canonical_document_bytes(first_document)
    assert len(first_document.translations) == 1
    assert {issue.details.get("legacy_group_id") for issue in first_document.issues} >= {"g2"}

    second = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True, dump_json=True
    )
    second_document = _open_document(config, state, page_id)
    assert second_document is not None

    assert first.status == second.status == "succeeded"
    assert calls == 1
    assert len(second_document.translations) == 1
    assert all(stage.cache_hit for stage in second_document.stages)
    assert first_bytes != canonical_document_bytes(second_document)
    assert (
        config.paths.output_dir / "debug" / "page_page_document.json"
    ).read_bytes() == canonical_document_bytes(second_document)


def test_resume_mutation_only_invalidates_affected_fingerprints(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls = 0

    def fake_process(path, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _legacy_result(path)

    monkeypatch.setattr(pipeline_module, "process_single_page", fake_process)
    first = pipeline_module.run_pipeline(config, job_id="job-1", state_dir=state, resume=True)
    changed = config.model_copy(
        update={
            "typesetting": config.typesetting.model_copy(
                update={"font_size_min": config.typesetting.font_size_min + 1}
            )
        }
    )
    pipeline_module.run_pipeline(changed, job_id="job-1", state_dir=state, resume=True)
    document = _open_document(config, state, first.pages[0].page_id)
    assert document is not None
    by_stage = {record.stage: record for record in document.stages}

    assert calls == 2
    assert by_stage[StageName.DETECT].cache_hit
    assert by_stage[StageName.OCR].cache_hit
    assert by_stage[StageName.TRANSLATE].cache_hit
    assert not by_stage[StageName.LAYOUT].cache_hit
    assert not by_stage[StageName.INPAINT_RENDER].cache_hit
    assert not by_stage[StageName.ENCODE].cache_hit


def test_inspect_and_replay_use_only_durable_state(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    source = _source(config)
    state = tmp_path / "state"
    monkeypatch.setattr(
        pipeline_module,
        "process_single_page",
        lambda path, *_args, **_kwargs: _legacy_result(path),
    )
    result = pipeline_module.run_pipeline(config, job_id="job-1", state_dir=state, resume=True)
    page_id = result.pages[0].page_id
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "openrouter:\n  api_key: unused\n  model: unused\n", encoding="utf-8"
    )
    manifest = tmp_path / "replayed.json"
    image = tmp_path / "replayed.png"
    runner = CliRunner()

    replay = runner.invoke(
        cli_module.cli,
        [
            "replay",
            "--config",
            str(config_path),
            "--state-dir",
            str(state),
            "--job",
            "job-1",
            "--page",
            page_id,
            "--output",
            str(manifest),
            "--output-image",
            str(image),
        ],
    )
    inspected = runner.invoke(
        cli_module.cli,
        [
            "inspect",
            "--config",
            str(config_path),
            "--state-dir",
            str(state),
            "--job",
            "job-1",
            "--page",
            page_id,
        ],
    )
    document = _open_document(config, state, page_id)

    assert replay.exit_code == 0, replay.output
    assert inspected.exit_code == 0, inspected.output
    assert document is not None
    assert manifest.read_bytes() == canonical_document_bytes(document)
    assert cv2.imread(str(image)) is not None
    assert '"active_regions"' in inspected.output
    assert '"cache_hit"' in inspected.output
    inspection = json.loads(inspected.output)
    assert str(state / "artifacts") in inspection["pages"][0]["document_artifact"]["path"]
    assert source.exists()
