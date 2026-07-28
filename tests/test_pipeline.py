from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_translator.config import AppConfig, PostprocessConfig
from manga_translator.detector import TextGroup, TextRegion
from manga_translator.pipeline import (
    _build_page_translation_units,
    _group_mask_iou,
    _merge_duplicate_groups,
    _merge_group_objects,
    _merge_translation_duplicates,
    _refresh_group_order,
    _resolve_render_collisions,
    get_image_files,
)


def group(
    group_id: str,
    bbox: tuple[int, int, int, int],
    text: str,
    *,
    confidence: float = 0.8,
    translation: str = "",
    translation_valid: bool = False,
    vertical: bool = True,
) -> TextGroup:
    _x, _y, w, h = bbox
    mask = np.full((h, w), 255, dtype=np.uint8)
    return TextGroup(
        id=group_id,
        region_ids=[f"r-{group_id}"],
        bbox=bbox,
        vertical=vertical,
        ocr_text=text,
        ocr_text_norm=text,
        ocr_confidence=confidence,
        translation=translation,
        translation_valid=translation_valid,
        status="ready" if translation_valid else "ocr_done",
        mask=mask,
    )


def test_image_pages_use_natural_sort(tmp_path: Path) -> None:
    for name in ("page10.png", "page2.png", "page1.png"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in get_image_files(tmp_path)] == [
        "page1.png",
        "page2.png",
        "page10.png",
    ]


def test_fuzzy_overlap_dedupes_same_dialogue_with_one_ocr_error() -> None:
    a = group("a", (10, 10, 80, 120), "今日はどうしたの")
    b = group("b", (14, 14, 76, 116), "今日はどうしたろ")
    merged = _merge_duplicate_groups([a, b], PostprocessConfig())
    assert len(merged) == 1
    assert set(merged[0].region_ids) == {"r-a", "r-b"}


def test_large_empty_container_does_not_swallow_small_real_bubbles() -> None:
    huge = group("huge", (0, 0, 1000, 1000), "", confidence=0.0)
    small_a = group("a", (40, 50, 80, 120), "一つ目")
    small_b = group("b", (700, 700, 80, 120), "二つ目")
    merged = _merge_duplicate_groups([huge, small_a, small_b], PostprocessConfig())
    assert len(merged) == 3


def test_separate_repeated_short_lines_are_not_deduped() -> None:
    a = group("a", (10, 10, 30, 50), "うん")
    b = group("b", (80, 10, 30, 50), "うん")
    merged = _merge_duplicate_groups([a, b], PostprocessConfig())
    assert len(merged) == 2


def test_translation_stage_catches_strong_overlap_duplicate() -> None:
    a = group(
        "a",
        (10, 10, 80, 120),
        "今日はどうしたの",
        translation="你今天怎麼了？",
        translation_valid=True,
    )
    b = group(
        "b",
        (13, 12, 78, 118),
        "今日どうしたの",
        translation="你今天怎麼了？",
        translation_valid=True,
    )
    merged = _merge_translation_duplicates([a, b], PostprocessConfig())
    assert len(merged) == 1
    assert merged[0].translation == "你今天怎麼了？"


def test_merged_group_mask_stays_local_not_page_sized() -> None:
    a = group("a", (10, 10, 20, 30), "a")
    b = group("b", (20, 20, 20, 30), "a")
    merged = _merge_group_objects(a, b)
    assert merged.mask is not None
    assert merged.mask.shape == (merged.h, merged.w)


def test_mask_iou_uses_local_overlap() -> None:
    a = group("a", (10, 10, 20, 20), "a")
    b = group("b", (20, 10, 20, 20), "a")
    assert _group_mask_iou(a, b) == 1 / 3


def test_auto_reading_order_is_not_reordered_before_translation() -> None:
    groups = [
        group("h1", (10, 10, 30, 20), "第一句", vertical=False),
        group("h2", (10, 100, 30, 20), "第二句", vertical=False),
        group("v", (500, 200, 20, 40), "第三句", vertical=True),
    ]
    ordered = _refresh_group_order(groups, {}, PostprocessConfig(reading_order="auto"))
    _same_groups, texts = _build_page_translation_units(ordered)

    assert texts == ["第一句", "第二句", "第三句"]


def test_geometry_collision_blocks_two_different_translations_at_same_position() -> None:
    primary = group(
        "a",
        (10, 10, 80, 120),
        "今日はどうしたの",
        translation="你今天怎麼了？",
        translation_valid=True,
    )
    fallback = group(
        "b",
        (13, 12, 78, 118),
        "別の幻覚文字",
        translation="完全不同的錯誤字幕",
        translation_valid=True,
    )
    regions = {
        "r-a": TextRegion(id="r-a", x=10, y=10, w=80, h=120, source="ctd"),
        "r-b": TextRegion(
            id="r-b",
            x=13,
            y=12,
            w=78,
            h=118,
            source="mask_fallback",
        ),
    }

    _resolve_render_collisions(
        [primary, fallback],
        regions,
        PostprocessConfig(),
    )

    assert primary.translation_valid
    assert not fallback.translation_valid
    assert fallback.status == "render_collision_rejected"


def test_batch_aborts_once_when_ocr_runtime_cannot_initialize(tmp_path, monkeypatch) -> None:
    import cv2
    import pytest

    from manga_translator import pipeline as pipeline_module
    from manga_translator.ocr import OCRInitializationError

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "page1.png"), np.full((16, 16, 3), 255, dtype=np.uint8))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
openrouter:
  api_key: test
  model: test/model
paths:
  input_dir: ./input
  output_dir: ./output
  glossary: ./missing-glossary.json
  font: ./missing-font.ttf
  font_fallback: ./missing-fallback.ttf
detection:
  model_path: ./missing-model.pt
  device: cpu
""".strip(),
        encoding="utf-8",
    )
    config = AppConfig.from_yaml(config_path)
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        raise OCRInitializationError("single startup failure")

    monkeypatch.setattr(pipeline_module, "initialize_ocr_model", fail_once)
    monkeypatch.setattr(pipeline_module, "process_single_page", lambda *args, **kwargs: None)

    with pytest.raises(OCRInitializationError, match="single startup failure"):
        pipeline_module.run_pipeline(config)

    assert calls == 1
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_mask_containment_blocks_nested_column_even_with_tiny_area_ratio() -> None:
    outer = group(
        "outer",
        (10, 10, 120, 220),
        "整句完整文字",
        confidence=0.90,
        translation="完整翻譯",
        translation_valid=True,
    )
    inner = group(
        "inner",
        (90, 30, 20, 120),
        "錯誤碎片",
        confidence=0.88,
        translation="錯誤重疊字幕",
        translation_valid=True,
    )
    regions = {
        "r-outer": TextRegion(id="r-outer", x=10, y=10, w=120, h=220, source="ctd"),
        "r-inner": TextRegion(
            id="r-inner",
            x=90,
            y=30,
            w=20,
            h=120,
            source="ctd_multiscale",
        ),
    }

    _resolve_render_collisions([outer, inner], regions, PostprocessConfig())

    assert outer.translation_valid
    assert not inner.translation_valid
    assert inner.status == "render_collision_rejected"


def test_planned_text_block_collision_detects_glyph_sized_overlap() -> None:
    from manga_translator.pipeline import _layout_blocks_conflict
    from manga_translator.typesetter import TextLayoutPlan

    a = TextLayoutPlan(
        bbox=(0, 0, 200, 200),
        direction="vertical",
        font_size=40,
        chunks=("甲乙",),
        primary_step=0,
        secondary_step=45,
        center_x=100,
        center_y=100,
        block_width=80,
        block_height=120,
    )
    b = TextLayoutPlan(
        bbox=(80, 40, 200, 200),
        direction="vertical",
        font_size=38,
        chunks=("丙丁",),
        primary_step=0,
        secondary_step=43,
        center_x=60,
        center_y=70,
        block_width=80,
        block_height=120,
    )
    far = TextLayoutPlan(
        bbox=(400, 400, 100, 100),
        direction="vertical",
        font_size=38,
        chunks=("戊",),
        primary_step=0,
        secondary_step=0,
        center_x=50,
        center_y=50,
        block_width=38,
        block_height=38,
    )

    assert _layout_blocks_conflict(a, b)
    assert not _layout_blocks_conflict(a, far)
