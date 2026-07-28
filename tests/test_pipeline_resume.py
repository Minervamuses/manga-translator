from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from manga_translator import cli as cli_module
from manga_translator import pipeline as pipeline_module
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.contracts.mapping import (
    MappingContractError,
    MappingIssue,
    RawResponseRef,
    ResponseItem,
    bind_validated_responses,
)
from manga_translator.detector import DetectionResult, MaskSource, TextGroup, TextRegion
from manga_translator.domain.issues import IssueCode, StageName, StageStatus
from manga_translator.domain.serialization import canonical_document_bytes
from manga_translator.manga_ocr_runtime import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from manga_translator.ocr import OCRCandidate, OCRResult
from manga_translator.storage import ArtifactStore, JobStore
from manga_translator.translator import TranslationValidation
from manga_translator.typesetter import TextLayoutPlan


def _config(tmp_path: Path) -> AppConfig:
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"テスト":"測試"}', encoding="utf-8")
    return AppConfig(
        openrouter=OpenRouterConfig(api_key="never-used", model="test/model"),
        paths=PathsConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            glossary=glossary,
            font=tmp_path / "font.ttf",
            font_fallback=tmp_path / "fallback.ttf",
        ),
    )


def _source(config: AppConfig) -> Path:
    path = config.paths.input_dir / "page.png"
    path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(path), np.full((40, 60, 3), 255, dtype=np.uint8))
    return path


def _detection() -> DetectionResult:
    regions = [
        TextRegion(
            id="r1",
            x=4,
            y=5,
            w=12,
            h=18,
            confidence=0.9,
            font_size_hint=18.0,
            page_bbox=(4.0, 5.0, 12.0, 18.0),
            line_polygons=(((5.0, 6.0), (14.0, 5.0), (15.0, 20.0), (6.0, 21.0)),),
            angle_degrees=6.5,
            local_mask=np.full((18, 12), 255, dtype=np.uint8),
            mask_source=MaskSource(
                detector_pass=0,
                detection_input_size=1024,
                raw_index=0,
                source="ctd",
                source_region_id="r1",
                page_to_local_affine=(1.0, 0.0, -4.0, 0.0, 1.0, -5.0),
            ),
        ),
        TextRegion(
            id="r2",
            x=34,
            y=5,
            w=12,
            h=18,
            confidence=0.8,
            page_bbox=(34.0, 5.0, 12.0, 18.0),
            local_mask=np.full((18, 12), 255, dtype=np.uint8),
            mask_source=MaskSource(
                detector_pass=0,
                detection_input_size=1024,
                raw_index=1,
                source="ctd",
                source_region_id="r2",
                page_to_local_affine=(1.0, 0.0, -34.0, 0.0, 1.0, -5.0),
            ),
        ),
    ]
    groups = [
        TextGroup(
            id="detected-1",
            region_ids=["r1"],
            bbox=(4, 5, 12, 18),
            vertical=True,
            mask=np.full((18, 12), 255, dtype=np.uint8),
        ),
        TextGroup(
            id="detected-2",
            region_ids=["r2"],
            bbox=(34, 5, 12, 18),
            vertical=True,
            mask=np.full((18, 12), 255, dtype=np.uint8),
        ),
    ]
    return DetectionResult(
        regions_raw=regions,
        regions_post=regions,
        groups=groups,
        mask=np.full((40, 60), 255, dtype=np.uint8),
        raw_mask=np.full((40, 60), 255, dtype=np.uint8),
    )


def _plan(group: TextGroup) -> TextLayoutPlan:
    x, y, width, height = group.bbox
    return TextLayoutPlan(
        bbox=group.bbox,
        direction="vertical",
        font_size=12,
        chunks=(group.translation,),
        primary_step=12.25,
        secondary_step=12.5,
        center_x=x + width / 2,
        center_y=y + height / 2,
        block_width=10.5,
        block_height=16.75,
    )


def _install_component_fakes(
    monkeypatch,
    config: AppConfig,
) -> tuple[Counter[str], list[bytes]]:
    calls: Counter[str] = Counter()
    provider_payloads: list[bytes] = []

    def detect(*_args, **_kwargs) -> DetectionResult:
        calls["detect"] += 1
        return _detection()

    def initialize() -> None:
        calls["ocr_initialize"] += 1

    def ocr(*, group, **_kwargs) -> OCRResult:
        calls["ocr"] += 1
        accepted = group.x < 20
        text = "テスト" if accepted else ""
        return OCRResult(
            text=text,
            normalized=text,
            confidence=0.98 if accepted else 0.0,
            source="fixture",
            candidates=[OCRCandidate(text, text, 0.98 if accepted else 0.0, "fixture")],
        )

    def assess(result, *_args, **_kwargs):
        return (bool(result.text), "" if result.text else "empty_fixture")

    def request(groups, page_id, _config, _glossary):
        calls["provider"] += 1
        _ordered, _texts, request_map = pipeline_module._build_translation_request(
            groups, page_id
        )
        raw = json.dumps(
            {
                "request_id": request_map.request_id,
                "translations": ["測試翻譯" for _item in request_map.items],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        provider_payloads.append(raw)
        relative = Path("translation-responses") / f"{hashlib.sha256(raw).hexdigest()}.json"
        path = config.paths.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        reference = RawResponseRef.from_bytes(
            raw,
            media_type="application/json",
            relative_path=relative.as_posix(),
        )
        return bind_validated_responses(
            request_map,
            [
                ResponseItem(
                    item_id=item.item_id,
                    source_sha256=item.source_sha256,
                    translation="測試翻譯",
                    response_index=index,
                    raw_response_ref=reference,
                )
                for index, item in enumerate(request_map.items)
            ],
        )

    def layout(_original, groups, _regions, _config):
        calls["layout"] += 1
        return {
            group.id: _plan(group)
            for group in groups
            if group.translation_valid and group.translation
        }

    def inpaint(original, _detection, _config):
        calls["inpaint"] += 1
        return original.copy()

    def render(*, image, group, **_kwargs):
        calls["render"] += 1
        result = image.copy()
        result[group.y, group.x] = (0, 0, 0)
        return result

    monkeypatch.setattr(pipeline_module, "detect_text_regions", detect)
    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", initialize)
    monkeypatch.setattr(pipeline_module, "ocr_group_detailed", ocr)
    monkeypatch.setattr(pipeline_module, "assess_ocr_result", assess)
    monkeypatch.setattr(pipeline_module, "_request_translations", request)
    monkeypatch.setattr(pipeline_module, "_preflight_layout_plans", layout)
    monkeypatch.setattr(pipeline_module, "inpaint_regions", inpaint)
    monkeypatch.setattr(pipeline_module, "render_text_into_group", render)
    monkeypatch.setattr(
        pipeline_module,
        "validate_translation",
        lambda *_args, **_kwargs: TranslationValidation(valid=True),
    )
    return calls, provider_payloads


def _open_document(state: Path, page_id: str):
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        return store.load_page_document(job_id="job-1", page_id=page_id)


def test_default_batch_and_single_page_entrypoints_persist_page_documents(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    source = _source(config)
    _install_component_fakes(monkeypatch, config)
    durable_root = config.paths.output_dir / ".manga-translator"

    batch = pipeline_module.run_pipeline(config)
    page_id = batch.pages[0].page_id
    single = pipeline_module.process_single_page(source, config, {})

    assert batch.status == single.status == "succeeded"
    assert not hasattr(pipeline_module, "_process_single_page_impl")
    assert (durable_root / "jobs.sqlite3").is_file()
    with JobStore(
        durable_root / "jobs.sqlite3", ArtifactStore(durable_root / "artifacts")
    ) as store:
        assert store.load_page_document(job_id="default", page_id=page_id) is not None
        assert store.load_page_document(job_id="single", page_id=page_id) is not None


def test_region_mask_artifacts_require_canonical_png_and_bbox_dimensions(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    bbox = pipeline_module.BoundingBox(x=1.0, y=2.0, width=12.0, height=18.0)
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        encoded_jpeg, jpeg = cv2.imencode(
            ".jpg", np.full((18, 12), 255, dtype=np.uint8)
        )
        assert encoded_jpeg
        jpeg_ref = store.store_artifact(
            jpeg.tobytes(),
            media_type="image/png",
            owner_type="test",
            owner_id="jpeg-mask",
        )
        with pytest.raises(ValueError, match="not canonical PNG"):
            pipeline_module._read_local_mask_artifact(store, jpeg_ref, bbox)

        encoded_png, png = cv2.imencode(
            ".png", np.full((17, 12), 255, dtype=np.uint8)
        )
        assert encoded_png
        wrong_shape_ref = store.store_artifact(
            png.tobytes(),
            media_type="image/png",
            owner_type="test",
            owner_id="wrong-shape-mask",
        )
        with pytest.raises(ValueError, match="do not match bbox"):
            pipeline_module._read_local_mask_artifact(store, wrong_shape_ref, bbox)


def test_component_stages_resume_without_reloading_models_or_provider(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls, provider_payloads = _install_component_fakes(monkeypatch, config)
    reconciliation_inputs = []
    real_reconcile = pipeline_module.reconcile_regions

    def reconcile_spy(**kwargs):
        reconciliation_inputs.append(kwargs)
        return real_reconcile(**kwargs)

    monkeypatch.setattr(pipeline_module, "reconcile_regions", reconcile_spy)

    first = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True, dump_json=True
    )
    first_calls = calls.copy()
    page_id = first.pages[0].page_id
    first_document = _open_document(state, page_id)
    assert first_document is not None
    first_bytes = canonical_document_bytes(first_document)
    assert len(first_document.translations) == 1
    assert first_document.region_revisions[0].angle_degrees == 6.5
    assert first_document.region_revisions[0].line_polygons
    assert first_document.region_revisions[0].mask_lineage[0].source_revision_id == (
        first_document.region_revisions[0].revision_id
    )
    assert len(first_document.group_geometries) == 2
    assert len(first_document.style_fingerprints) == 2
    assert all(
        fingerprint.source == "original_image"
        for fingerprint in first_document.style_fingerprints
    )

    second = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True, dump_json=True
    )
    second_document = _open_document(state, page_id)

    assert first.status == second.status == "succeeded"
    assert calls == first_calls
    assert {group.id for group in second.pages[0].groups if group.translation_valid} == {
        "g001"
    }
    assert {group.id for group in second.pages[0].groups if not group.translation_valid} == {
        "g000"
    }
    assert second_document is not None
    assert canonical_document_bytes(second_document) == first_bytes
    assert all(
        observation.crop_dhash is not None and observation.mask is not None
        for call in reconciliation_inputs
        for observation in call["observations"]
    )
    assert reconciliation_inputs[-1]["previous_masks"]
    assert reconciliation_inputs[-1]["previous_crop_dhashes"]
    assert all(revision.mask_refs for revision in second_document.region_revisions)
    assert all(
        record.model_revision == f"{DEFAULT_MODEL_ID}:{DEFAULT_MODEL_REVISION}"
        for record in second_document.ocr_records
    )
    assert len(second_document.translations) == 1
    raw_ref = second_document.translations[0].raw_response_ref
    assert ArtifactStore(state / "artifacts").read_bytes(raw_ref.sha256) == provider_payloads[0]
    assert len(second.pages[0].mapping_chains) == 2
    assert all(
        set(snapshot.chain)
        == {
            "region",
            "ocr_record",
            "request_item",
            "raw_response_item",
            "validated_translation",
            "layout_plan",
            "render_target",
        }
        for snapshot in second.pages[0].mapping_chains
    )
    translated_mapping = next(
        snapshot
        for snapshot in second.pages[0].mapping_chains
        if snapshot.translation_valid
    )
    assert all(UUID(region_id) for region_id in translated_mapping.region_ids)
    assert isinstance(translated_mapping.chain["region"], dict)
    assert translated_mapping.chain["region"]["revision_ids"]
    for key in ("layout_plan", "render_target"):
        artifact = translated_mapping.chain[key]["artifact"]
        assert ArtifactStore(state / "artifacts").exists(
            artifact["sha256"], expected_size=artifact["size_bytes"]
        )


def test_font_and_glossary_mutations_invalidate_only_true_downstream_components(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls, _provider_payloads = _install_component_fakes(monkeypatch, config)
    pipeline_module.run_pipeline(config, job_id="job-1", state_dir=state, resume=True)
    before_font = calls.copy()

    font_changed = config.model_copy(
        update={
            "typesetting": config.typesetting.model_copy(
                update={"font_size_min": config.typesetting.font_size_min + 1}
            )
        }
    )
    pipeline_module.run_pipeline(
        font_changed, job_id="job-1", state_dir=state, resume=True
    )

    assert calls["detect"] == before_font["detect"]
    assert calls["ocr"] == before_font["ocr"]
    assert calls["provider"] == before_font["provider"]
    assert calls["layout"] == before_font["layout"] + 1
    assert calls["inpaint"] == before_font["inpaint"] + 1
    assert calls["render"] == before_font["render"] + 1

    before_glossary = calls.copy()
    config.paths.glossary.write_text('{"テスト":"試驗"}', encoding="utf-8")
    pipeline_module.run_pipeline(
        font_changed, job_id="job-1", state_dir=state, resume=True
    )

    assert calls["detect"] == before_glossary["detect"]
    assert calls["ocr"] == before_glossary["ocr"]
    assert calls["provider"] == before_glossary["provider"] + 1
    assert calls["layout"] == before_glossary["layout"] + 1


def test_crash_after_provider_bundle_does_not_fetch_again(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls, _provider_payloads = _install_component_fakes(monkeypatch, config)
    real_apply = pipeline_module._apply_translation_batch
    crashed = False

    def crash_once(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated crash after provider bundle")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_apply_translation_batch", crash_once)
    first = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )
    second = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )

    assert first.status == "failed"
    assert second.status == "succeeded"
    assert calls["provider"] == 1


def test_rejected_provider_response_is_replayed_and_blocks_downstream(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _source(config)
    state = tmp_path / "state"
    calls, _provider_payloads = _install_component_fakes(monkeypatch, config)
    raw = b'{"provider":"malformed mapped response"}'
    raw_sha256 = hashlib.sha256(raw).hexdigest()

    def rejected(groups, page_id, _config, _glossary):
        calls["provider"] += 1
        _ordered, _texts, request_map = pipeline_module._build_translation_request(
            groups, page_id
        )
        relative = Path("translation-responses") / f"{raw_sha256}.json"
        path = config.paths.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        reference = RawResponseRef.from_bytes(
            raw,
            media_type="application/json",
            relative_path=relative.as_posix(),
        )
        raise MappingContractError(
            [MappingIssue("missing_id", {"ids": [request_map.items[0].item_id]})],
            raw_response_refs=[reference],
        )

    monkeypatch.setattr(pipeline_module, "_request_translations", rejected)
    first = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )
    second = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )

    assert first.status == second.status == "failed"
    assert first.pages[0].stage_failure == second.pages[0].stage_failure == "translate"
    assert calls["provider"] == 1
    assert calls["layout"] == calls["inpaint"] == calls["render"] == 0
    assert ArtifactStore(state / "artifacts").read_bytes(raw_sha256) == raw
    page_id = first.pages[0].page_id
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        statuses = {
            row[0]: row[2]
            for row in store.list_stage_runs(job_id="job-1", page_id=page_id)
        }
    assert statuses["translate"] == "failed"
    document = _open_document(state, page_id)
    assert document is not None
    translate_record = next(
        record for record in document.stages if record.stage is StageName.TRANSLATE
    )
    assert translate_record.status is StageStatus.FAILED
    assert translate_record.issues[0].code is IssueCode.TRANSLATION_FAILED


def test_inspect_and_replay_need_only_durable_state(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    source = _source(config)
    state = tmp_path / "state"
    _install_component_fakes(monkeypatch, config)
    result = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )
    page_id = result.pages[0].page_id
    resumed = pipeline_module.run_pipeline(
        config, job_id="job-1", state_dir=state, resume=True
    )
    assert resumed.status == "succeeded"
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        document = store.load_page_document(job_id="job-1", page_id=page_id)
        assert document is not None
        identities = (
            document.region_identities[0].model_copy(update={"is_active": False}),
            *document.region_identities[1:],
        )
        store.store_page_document(
            "job-1", document.model_copy(update={"region_identities": identities})
        )
    source.unlink()
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
    document = _open_document(state, page_id)

    assert replay.exit_code == 0, replay.output
    assert inspected.exit_code == 0, inspected.output
    assert document is not None
    assert manifest.read_bytes() == canonical_document_bytes(document)
    assert cv2.imread(str(image)) is not None
    inspection = json.loads(inspected.output)
    inspected_page = inspection["pages"][0]
    assert len(inspected_page["active_regions"]) == 1
    assert all(stage["cache_hit"] for stage in inspected_page["stages"])
    assert str(state / "artifacts") in inspected_page["document_artifact"]["path"]
