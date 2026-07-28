from __future__ import annotations

import hashlib

import numpy as np

from manga_translator.artifacts import _group_to_dict
from manga_translator.config import AppConfig, InpaintingConfig, OpenRouterConfig
from manga_translator.contracts.mapping import (
    MappingContractError,
    MappingIssue,
    RawResponseRef,
    bind_validated_values,
    build_request_map,
)
from manga_translator.detector import DetectionResult, TextGroup
from manga_translator.inpainter import inpaint_regions
from manga_translator.pipeline import (
    _mapping_snapshots,
    _merge_translation_duplicates,
    _record_mapping_layout_plans,
    _translate_groups,
)
from manga_translator.result import GroupMappingSnapshot
from manga_translator.translator import ProviderResponseError


def _config() -> AppConfig:
    return AppConfig(openrouter=OpenRouterConfig(api_key="test", model="test/model"))


def _group(group_id: str, source: str) -> TextGroup:
    return TextGroup(
        id=group_id,
        region_ids=[],
        bbox=(0, 0, 10, 10),
        vertical=True,
        ocr_text=source,
        ocr_text_norm=source,
        ocr_confidence=1.0,
        status="ocr_done",
        mask=np.full((10, 10), 255, dtype=np.uint8),
    )


def test_pipeline_maps_translations_by_region_key_not_batch_position(monkeypatch) -> None:
    first = _group("g-a", "猫だ")
    second = _group("g-b", "犬だ")

    def reversed_batch(groups, page_id, _config, _glossary):
        request = build_request_map(
            page_id,
            [
                (groups[1].mapping_region_key, groups[1].ocr_text),
                (groups[0].mapping_region_key, groups[0].ocr_text),
            ],
        )
        return bind_validated_values(request, ["是狗", "||是貓..."])

    monkeypatch.setattr("manga_translator.pipeline._request_translations", reversed_batch)

    _translate_groups([first, second], "page", _config(), {})

    assert first.translation == "是貓"
    assert second.translation == "是狗"
    assert first.mapping_chain["region"] == first.mapping_region_key
    assert first.mapping_chain["validated_translation"] == hashlib.sha256(
        first.translation.encode("utf-8")
    ).hexdigest()
    debug_chain = _group_to_dict(first)["mapping_chain"]
    normal_chain = GroupMappingSnapshot.from_group(first).to_manifest()["chain"]
    assert debug_chain["raw_response_item"]
    assert debug_chain == normal_chain


def test_translation_rejection_keeps_prefix_and_nulls_unvalidated_stages(monkeypatch) -> None:
    rejected = _group("g-rejected", "こんにちは")

    def echoed_batch(groups, page_id, _config, _glossary):
        request = build_request_map(
            page_id,
            [(groups[0].mapping_region_key, groups[0].ocr_text)],
        )
        return bind_validated_values(request, [groups[0].ocr_text])

    monkeypatch.setattr("manga_translator.pipeline._request_translations", echoed_batch)

    _translate_groups([rejected], "page", _config(), {})

    assert rejected.status == "translation_rejected"
    assert not rejected.translation_valid
    assert rejected.mapping_chain["region"] == rejected.mapping_region_key
    assert rejected.mapping_chain["ocr_record"]
    assert rejected.mapping_chain["request_item"]
    assert rejected.mapping_chain["raw_response_item"]
    assert rejected.mapping_chain["validated_translation"] is None
    assert rejected.mapping_chain["layout_plan"] is None
    assert rejected.mapping_chain["render_target"] is None


def test_ocr_failures_keep_completed_mapping_prefix() -> None:
    rejected = _group("g-rejected", "")
    rejected.ocr_text_norm = ""
    rejected.status = "ocr_rejected"
    failed = _group("g-failed", "")
    failed.ocr_text_norm = ""
    failed.status = "ocr_failed"

    issue = _translate_groups([rejected, failed], "page", _config(), {})

    assert issue is None
    for group in (rejected, failed):
        assert group.mapping_chain["region"] == group.mapping_region_key
        assert group.mapping_chain["ocr_record"] == f"ocr:{group.mapping_region_key}"
        assert group.mapping_chain["request_item"] is None
        assert group.mapping_chain["raw_response_item"] is None
        assert group.mapping_chain["validated_translation"] is None
        assert group.mapping_chain["layout_plan"] is None
        assert group.mapping_chain["render_target"] is None


def test_request_map_initialization_failure_is_a_typed_translation_failure() -> None:
    first = _group("g-a", "猫だ")
    second = _group("g-b", "犬だ")
    first.mapping_region_key = "group:duplicate"
    second.mapping_region_key = "group:duplicate"

    issue = _translate_groups([first, second], "page", _config(), {})

    assert issue is not None
    assert issue.code == "translation_mapping_failed"
    assert all(group.status == "translation_failed" for group in (first, second))
    assert all(group.mapping_chain["region"] == "group:duplicate" for group in (first, second))
    assert all(group.mapping_chain["ocr_record"] for group in (first, second))
    assert all(group.mapping_chain["request_item"] is None for group in (first, second))


def test_layout_rejection_records_plan_but_not_render_target() -> None:
    rejected = _group("g-layout", "こんにちは")
    rejected.translation = "你好"
    rejected.translation_valid = False
    rejected.status = "layout_rejected"
    rejected.layout_info = {"fits": False, "reason": "too_small"}
    rejected.mapping_chain = {
        "region": "group:layout",
        "ocr_record": "ocr:group:layout",
        "request_item": "R-layout:T0000",
        "raw_response_item": {"item_id": "R-layout:T0000"},
        "validated_translation": hashlib.sha256("你好".encode()).hexdigest(),
        "layout_plan": None,
        "render_target": None,
    }

    _record_mapping_layout_plans([rejected])

    assert rejected.mapping_chain["layout_plan"] == "layout:g-layout"
    assert rejected.mapping_chain["render_target"] is None


def test_post_translation_merge_preserves_each_request_mapping_outcome() -> None:
    first = _group("g-a", "猫だ")
    second = _group("g-b", "猫です")
    first.mapping_region_key = "group:first"
    second.mapping_region_key = "group:second"
    second.ocr_confidence = 1.1
    request = build_request_map(
        "page",
        [
            (first.mapping_region_key, first.ocr_text),
            (second.mapping_region_key, second.ocr_text),
        ],
    )
    translations = bind_validated_values(request, ["這是貓咪", "這是貓咪"])
    for group in (first, second):
        group.translation = "這是貓咪"
        group.translation_valid = True
        group.status = "ready"
        group.mapping_chain = translations.chain_for(group.mapping_region_key)

    request_groups = [first, second]
    final_groups = _merge_translation_duplicates(request_groups, _config().postprocess)
    assert len(final_groups) == 1
    assert final_groups[0].id == second.id
    assert final_groups[0].mapping_chain["request_item"] == request.items[1].item_id
    final_groups[0].mapping_chain["layout_plan"] = f"layout:{final_groups[0].id}"
    final_groups[0].mapping_chain["render_target"] = f"render:{final_groups[0].id}"

    snapshots = _mapping_snapshots(request_groups, final_groups)

    assert len(snapshots) == 2
    by_item = {snapshot.chain["request_item"]: snapshot for snapshot in snapshots}
    assert set(by_item) == {item.item_id for item in request.items}
    assert sum(snapshot.chain["render_target"] is not None for snapshot in snapshots) == 1
    assert sum(snapshot.duplicate_of is not None for snapshot in snapshots) == 1
    assert by_item[request.items[0].item_id].duplicate_of == second.id
    assert by_item[request.items[1].item_id].duplicate_of is None


def test_mapping_failure_invalidates_entire_batch_and_produces_zero_inpaint_mask(
    monkeypatch,
) -> None:
    groups = [_group("g-a", "猫だ"), _group("g-b", "犬だ")]
    raw_reference = RawResponseRef.from_bytes(
        b'{"invalid":"response"}',
        media_type="application/json",
        relative_path="artifacts/translation-responses/invalid.json",
    )

    def reject(*_args, **_kwargs):
        raise MappingContractError(
            [MappingIssue("duplicate_id", {})],
            raw_response_refs=[raw_reference],
        )

    monkeypatch.setattr("manga_translator.pipeline._request_translations", reject)
    issue = _translate_groups(groups, "page", _config(), {})

    assert issue is not None
    assert issue.details["raw_response_artifacts"] == [raw_reference.to_dict()]
    assert all(not group.translation_valid and not group.translation for group in groups)
    assert all(group.mapping_chain["request_item"] for group in groups)
    assert all(group.mapping_chain["raw_response_item"] is None for group in groups)
    assert all(group.mapping_chain["validated_translation"] is None for group in groups)
    detection = DetectionResult([], [], groups, np.zeros((10, 10), dtype=np.uint8))
    source = np.full((10, 10, 3), 80, dtype=np.uint8)
    result = inpaint_regions(
        source,
        detection,
        InpaintingConfig(method="white", mask_dilate=0, extra_mask_dilate=0),
    )
    assert np.array_equal(result, source)


def test_provider_parse_failure_links_raw_artifact_from_page_issue(monkeypatch) -> None:
    group = _group("g-a", "猫だ")
    raw_reference = RawResponseRef.from_bytes(
        b"not-json",
        media_type="application/json",
        relative_path="artifacts/translation-responses/not-json.json",
    )

    def reject(*_args, **_kwargs):
        raise ProviderResponseError("invalid provider response", [raw_reference])

    monkeypatch.setattr("manga_translator.pipeline._request_translations", reject)

    issue = _translate_groups([group], "page", _config(), {})

    assert issue is not None
    assert issue.code == "translation_api_failed"
    assert issue.details["raw_response_artifacts"] == [raw_reference.to_dict()]
    assert group.mapping_chain["raw_response_item"] is None
