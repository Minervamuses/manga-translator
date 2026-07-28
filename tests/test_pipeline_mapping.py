from __future__ import annotations

import numpy as np

from manga_translator.artifacts import _group_to_dict
from manga_translator.config import AppConfig, InpaintingConfig, OpenRouterConfig
from manga_translator.contracts.mapping import (
    MappingContractError,
    MappingIssue,
    bind_validated_values,
    build_request_map,
)
from manga_translator.detector import DetectionResult, TextGroup
from manga_translator.inpainter import inpaint_regions
from manga_translator.pipeline import _translate_groups


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
        return bind_validated_values(request, ["是狗", "是貓"])

    monkeypatch.setattr("manga_translator.pipeline._request_translations", reversed_batch)

    _translate_groups([first, second], "page", _config(), {})

    assert first.translation == "是貓"
    assert second.translation == "是狗"
    assert first.mapping_chain["region"] == first.mapping_region_key
    assert _group_to_dict(first)["mapping_chain"]["raw_response_item"]


def test_mapping_failure_invalidates_entire_batch_and_produces_zero_inpaint_mask(
    monkeypatch,
) -> None:
    groups = [_group("g-a", "猫だ"), _group("g-b", "犬だ")]

    def reject(*_args, **_kwargs):
        raise MappingContractError([MappingIssue("duplicate_id", {})])

    monkeypatch.setattr("manga_translator.pipeline._request_translations", reject)
    _translate_groups(groups, "page", _config(), {})

    assert all(not group.translation_valid and not group.translation for group in groups)
    detection = DetectionResult([], [], groups, np.zeros((10, 10), dtype=np.uint8))
    source = np.full((10, 10, 3), 80, dtype=np.uint8)
    result = inpaint_regions(
        source,
        detection,
        InpaintingConfig(method="white", mask_dilate=0, extra_mask_dilate=0),
    )
    assert np.array_equal(result, source)
