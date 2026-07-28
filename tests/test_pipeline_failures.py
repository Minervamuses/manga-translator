from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from manga_translator import cli as cli_module
from manga_translator import pipeline as pipeline_module
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.detector import DetectionResult, TextGroup, TextRegion
from manga_translator.image_io import ImageEncodeError, ImageWriteError
from manga_translator.pipeline import _translate_groups
from manga_translator.result import (
    BatchResult,
    GroupMappingSnapshot,
    PageResult,
    ResultIssue,
)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        openrouter=OpenRouterConfig(api_key="test", model="test/model"),
        paths=PathsConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            glossary=tmp_path / "missing-glossary.json",
            font=tmp_path / "missing-font.ttf",
            font_fallback=tmp_path / "missing-fallback.ttf",
        ),
    )


def _write_source(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((12, 12, 3), value, dtype=np.uint8))


def _success_page(path: Path, page_id: str) -> PageResult:
    return PageResult(
        page_id=page_id,
        source_path=path,
        status="succeeded",
        image=np.full((12, 12, 3), 127, dtype=np.uint8),
    )


def _failed_page(path: Path, page_id: str) -> PageResult:
    return PageResult(
        page_id=page_id,
        source_path=path,
        status="blocked",
        issues=[
            ResultIssue(
                code="translation_api_failed",
                message="provider unavailable",
                stage="translation",
                page_id=page_id,
            )
        ],
        stage_failure="translation",
    )


def test_process_single_page_returns_typed_result_with_content_page_id(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    source = config.paths.input_dir / "blank.png"
    _write_source(source)
    monkeypatch.setattr(
        pipeline_module,
        "detect_text_regions",
        lambda image, *_args: DetectionResult(
            regions_raw=[],
            regions_post=[],
            groups=[],
            mask=np.zeros(image.shape[:2], dtype=np.uint8),
        ),
    )

    result = pipeline_module.process_single_page(source, config, {})

    assert isinstance(result, PageResult)
    assert result.status == "succeeded"
    assert result.page_id == hashlib.sha256(source.read_bytes()).hexdigest()


def test_second_page_failure_is_isolated_and_manifest_is_partial(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    first = config.paths.input_dir / "page1.png"
    second = config.paths.input_dir / "page2.png"
    _write_source(first, 240)
    _write_source(second, 220)

    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "process_single_page",
        lambda image_path, *_args, **_kwargs: (
            _success_page(image_path, "page-id-1")
            if image_path.name == "page1.png"
            else _failed_page(image_path, "page-id-2")
        ),
    )

    result = pipeline_module.run_pipeline(config)

    failed_output = config.paths.output_dir / "failed" / "page2.source-preserved.png"
    assert result.status == "partial"
    assert (config.paths.output_dir / "page1.png").is_file()
    assert not (config.paths.output_dir / "page2.png").exists()
    assert failed_output.read_bytes() == second.read_bytes()
    manifest = json.loads((config.paths.output_dir / "batch-manifest.json").read_text("utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["partial"] is True
    assert [page["page_id"] for page in manifest["pages"]] == ["page-id-1", "page-id-2"]
    assert manifest["pages"][1]["source_preserved"] is True
    assert manifest["pages"][1]["output_path"] == str(failed_output)


def test_normal_batch_manifest_publishes_traceable_mapping_chain_without_debug(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    source = config.paths.input_dir / "page.png"
    _write_source(source)
    raw_response = b'{"provider":"response"}'
    raw_hash = hashlib.sha256(raw_response).hexdigest()
    relative_path = f"artifacts/translation-responses/{raw_hash}.json"
    artifact_path = config.paths.output_dir / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(raw_response)
    group = TextGroup(
        id="g001",
        region_ids=["r001"],
        bbox=(1, 1, 8, 8),
        vertical=True,
        translation="你好",
        translation_valid=True,
        status="ready",
        mapping_region_key="group:stable",
        mapping_chain={
            "region": "group:stable",
            "ocr_record": "ocr:group:stable",
            "request_item": "R-test:T0000",
            "raw_response_item": {
                "item_id": "R-test:T0000",
                "response_index": 0,
                "artifact": {
                    "sha256": raw_hash,
                    "media_type": "application/json",
                    "size_bytes": len(raw_response),
                    "relative_path": relative_path,
                },
            },
            "validated_translation": hashlib.sha256("你好".encode()).hexdigest(),
            "layout_plan": "layout:g001",
            "render_target": "render:g001",
        },
    )
    page = _success_page(source, "page-id")
    page.mapping_chains = [GroupMappingSnapshot.from_group(group)]
    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "process_single_page",
        lambda *_args, **_kwargs: page,
    )

    result = pipeline_module.run_pipeline(config)

    assert result.status == "succeeded"
    manifest = json.loads((config.paths.output_dir / "batch-manifest.json").read_text("utf-8"))
    mapping = manifest["pages"][0]["mapping_chains"][0]
    assert mapping["group_id"] == "g001"
    assert mapping["group_status"] == "ready"
    assert mapping["translation_valid"] is True
    assert list(mapping["chain"]) == [
        "region",
        "ocr_record",
        "request_item",
        "raw_response_item",
        "validated_translation",
        "layout_plan",
        "render_target",
    ]
    artifact = mapping["chain"]["raw_response_item"]["artifact"]
    persisted = config.paths.output_dir / artifact["relative_path"]
    assert persisted.read_bytes() == raw_response
    assert hashlib.sha256(persisted.read_bytes()).hexdigest() == artifact["sha256"]
    assert not (config.paths.output_dir / "debug").exists()


def test_allow_partial_changes_only_cli_exit_code(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "openrouter:\n  api_key: test\n  model: test/model\n",
        encoding="utf-8",
    )
    page = _failed_page(tmp_path / "page2.png", "page-id-2")
    batch = BatchResult(status="partial", pages=[_success_page(tmp_path / "page1.png", "1"), page])
    calls: list[dict[str, object]] = []

    def return_partial(*_args, **kwargs):
        calls.append(kwargs)
        return batch

    monkeypatch.setattr(cli_module, "run_pipeline", return_partial)
    runner = CliRunner()

    default = runner.invoke(cli_module.cli, ["run", "--config", str(config_path)])
    allowed = runner.invoke(
        cli_module.cli,
        ["run", "--config", str(config_path), "--allow-partial"],
    )

    assert default.exit_code == 1
    assert allowed.exit_code == 0
    assert calls[0] == calls[1]


def test_no_input_files_has_distinct_issue_and_non_success_result(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.paths.input_dir.mkdir()
    monkeypatch.setattr(
        pipeline_module,
        "initialize_ocr_model",
        lambda: pytest.fail("OCR must not initialize without input"),
    )

    result = pipeline_module.run_pipeline(config)

    assert result.status == "failed"
    assert result.issues[0].code == "no_input_files"
    assert result.manifest_path is not None


def test_unreadable_image_uses_read_issue_and_failed_namespace(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    source = config.paths.input_dir / "broken.png"
    source.parent.mkdir()
    source.write_bytes(b"not an image")
    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", lambda: None)

    result = pipeline_module.run_pipeline(config)

    page = result.pages[0]
    assert page.status == "failed"
    assert page.issues[0].code == "image_read_failed"
    assert page.source_preserved
    assert page.output_path == config.paths.output_dir / "failed" / "broken.source-preserved.png"
    assert not (config.paths.output_dir / "broken.png").exists()


@pytest.mark.parametrize(
    ("error", "issue_code"),
    [
        (ImageEncodeError("encoder rejected image"), "image_encode_failed"),
        (ImageWriteError("disk rejected bytes"), "output_write_failed"),
    ],
)
def test_encode_and_output_write_failures_have_distinct_codes(
    tmp_path,
    monkeypatch,
    error,
    issue_code,
) -> None:
    config = _config(tmp_path)
    source = config.paths.input_dir / "page.png"
    _write_source(source)
    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "process_single_page",
        lambda image_path, *_args, **_kwargs: _success_page(image_path, "page-id"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_image_or_raise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    result = pipeline_module.run_pipeline(config)

    page = result.pages[0]
    assert page.issues[0].code == issue_code
    assert page.source_preserved
    assert not (config.paths.output_dir / "page.png").exists()


def test_api_failure_has_distinct_issue_and_preserves_group_text(monkeypatch) -> None:
    config = _config(Path("."))
    group = TextGroup(
        id="g1",
        region_ids=["r1"],
        bbox=(0, 0, 10, 10),
        vertical=True,
        ocr_text="猫だ",
        ocr_text_norm="猫だ",
        ocr_confidence=1.0,
        status="ocr_done",
        mask=np.full((10, 10), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_request_translations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )

    issue = _translate_groups([group], "page-id", config, {})

    assert issue is not None and issue.code == "translation_api_failed"
    assert group.status == "translation_failed"
    assert not group.translation_valid
    assert group.translation == ""


def test_group_ocr_exception_is_blocked_and_preserves_source(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    source = config.paths.input_dir / "page.png"
    _write_source(source, 231)
    region = TextRegion(id="r1", x=1, y=1, w=8, h=8, source="ctd")
    group = TextGroup(
        id="g000",
        region_ids=[region.id],
        bbox=(1, 1, 8, 8),
        vertical=True,
        mask=np.full((8, 8), 255, dtype=np.uint8),
    )
    detection = DetectionResult(
        regions_raw=[region],
        regions_post=[region],
        groups=[group],
        mask=np.full((12, 12), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", lambda: None)
    monkeypatch.setattr(pipeline_module, "detect_text_regions", lambda *_args: detection)
    monkeypatch.setattr(
        pipeline_module,
        "ocr_group_detailed",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("region OCR exploded")),
    )
    monkeypatch.setattr(pipeline_module, "_preflight_layout_plans", lambda *_args: {})

    result = pipeline_module.run_pipeline(config)

    assert result.status == "blocked"
    page = result.pages[0]
    assert page.status == "blocked"
    assert page.stage_failure == "ocr"
    assert page.source_preserved
    assert page.output_path == config.paths.output_dir / "failed" / "page.source-preserved.png"
    assert page.output_path.read_bytes() == source.read_bytes()
    assert not (config.paths.output_dir / "page.png").exists()
    assert group.status == "ocr_failed"
    assert not group.translation_valid
    assert len(page.issues) == 1
    assert page.issues[0].code == "ocr_group_failed"
    assert page.issues[0].stage == "ocr"
    assert page.issues[0].details == {
        "group_id": "g000",
        "group_status": "ocr_failed",
        "reason": "region OCR exploded",
        "region_ids": ["r1"],
    }
    manifest = json.loads((config.paths.output_dir / "batch-manifest.json").read_text("utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["pages"][0]["issues"][0]["message"] == "region OCR exploded"


@pytest.mark.parametrize(
    ("status", "issue_code", "stage"),
    [
        ("ocr_failed", "ocr_group_failed", "ocr"),
        ("translation_rejected", "translation_rejected", "translation"),
        ("layout_rejected", "layout_rejected", "layout"),
        ("layout_collision_rejected", "layout_collision_rejected", "layout"),
    ],
)
def test_group_failures_have_typed_issues(status, issue_code, stage) -> None:
    group = TextGroup(
        id="g007",
        region_ids=["r4", "r5"],
        bbox=(0, 0, 10, 10),
        vertical=True,
        status=status,
        skip_reason="specific failure",
    )

    issues = pipeline_module._group_failure_issues([group], "page-id")

    assert len(issues) == 1
    assert issues[0].code == issue_code
    assert issues[0].stage == stage
    assert issues[0].page_id == "page-id"
    assert issues[0].details == {
        "group_id": "g007",
        "group_status": status,
        "reason": "specific failure",
        "region_ids": ["r4", "r5"],
    }


@pytest.mark.parametrize("status", ["ocr_rejected", "render_collision_rejected", "ready"])
def test_expected_group_filtering_does_not_create_failure_issue(status) -> None:
    group = TextGroup(
        id="g1",
        region_ids=["r1"],
        bbox=(0, 0, 10, 10),
        vertical=True,
        status=status,
        skip_reason="expected filtering",
    )

    assert pipeline_module._group_failure_issues([group], "page-id") == []
