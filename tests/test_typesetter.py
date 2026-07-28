from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_translator.config import TypesettingConfig
from manga_translator.detector import TextGroup, TextRegion
from manga_translator.typesetter import (
    _build_group_local_mask,
    _calculate_font_size,
    _get_font_and_char,
    _has_glyph,
    _sanitize_render_text,
    _tight_layout_bbox,
    compose_patch_back,
    render_text_into_patch,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = str((ROOT / "fonts/Iansui-Regular.ttf").resolve())
FALLBACK = str((ROOT / "fonts/NotoSansCJKtc-Regular.otf").resolve())


def test_missing_glyph_is_not_mistaken_for_font_notdef_box() -> None:
    assert _has_glyph(PRIMARY, 28, "中")
    assert not _has_glyph(PRIMARY, 28, "\u0378")


def test_unsupported_character_uses_visible_replacement() -> None:
    _font, draw_char = _get_font_and_char("😀", 28, PRIMARY, FALLBACK, True)
    assert draw_char in {"□", "?", "・"}


def test_render_sanitizes_replacement_and_private_use_characters() -> None:
    assert _sanitize_render_text("你\ufffd好\ue000") == "你好"


def test_render_with_unsupported_glyph_does_not_crash_or_emit_empty_layer() -> None:
    patch = np.full((100, 160, 3), 255, dtype=np.uint8)
    layer = render_text_into_patch(
        patch,
        "你好😀",
        direction="horizontal",
        font_path=PRIMARY,
        fallback_font_path=FALLBACK,
        cfg=TypesettingConfig(font_size_min=18, font_size_max=28),
    )
    assert layer.shape == (100, 160, 4)
    assert int(np.count_nonzero(layer[:, :, 3])) > 0


def test_patch_composition_only_changes_target_roi() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    patch = np.zeros((6, 8, 4), dtype=np.uint8)
    patch[:, :, :3] = (255, 0, 0)  # RGBA red
    patch[:, :, 3] = 255

    result = compose_patch_back(image, patch, 5, 7)

    assert np.all(result[7:13, 5:13] == (0, 0, 255))  # BGR red
    assert np.count_nonzero(result[:7]) == 0
    assert np.count_nonzero(result[13:]) == 0


def test_group_mask_uses_real_pixels_not_region_rectangle() -> None:
    local = np.zeros((20, 30), dtype=np.uint8)
    local[7:11, 9:14] = 255
    region = TextRegion(id="r0", x=40, y=50, w=30, h=20, local_mask=local)
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=False,
        mask=local,
    )

    mask = _build_group_local_mask(group, {"r0": region}, group.bbox)

    assert int(np.count_nonzero(mask)) == 20


def test_layout_bbox_is_tight_around_original_text_pixels() -> None:
    local = np.zeros((40, 60), dtype=np.uint8)
    local[10:30, 20:35] = 255
    group = TextGroup(
        id="g0",
        region_ids=[],
        bbox=(100, 80, 60, 40),
        vertical=True,
        mask=local,
    )
    cfg = TypesettingConfig(layout_mask_dilate=0, layout_padding_px=2)

    assert _tight_layout_bbox(group, {}, cfg, (300, 300)) == (118, 88, 19, 24)


def test_detector_font_hint_caps_short_translation_growth() -> None:
    cfg = TypesettingConfig(font_size_min=10, font_size_max=80, max_font_growth_ratio=1.05)
    size = _calculate_font_size(
        "你好",
        200,
        200,
        PRIMARY,
        "horizontal",
        cfg,
        preferred_font_size=24,
    )
    assert size <= 26


def test_mask_projection_rejects_extreme_detector_font_hint() -> None:
    from manga_translator.typesetter import _preferred_group_font_size

    local = np.zeros((260, 140), dtype=np.uint8)
    local[20:240, 18:58] = 255
    local[20:240, 82:122] = 255
    region = TextRegion(
        id="r0",
        x=50,
        y=40,
        w=140,
        h=260,
        source="ctd",
        font_size_hint=138,
        local_mask=local,
    )
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=True,
        ocr_text="今日は学校に潜入している",
        mask=local,
    )

    preferred = _preferred_group_font_size(group, {"r0": region})

    assert preferred is not None
    assert 30 <= preferred <= 70


def test_preserve_layout_keeps_near_original_font_and_reuses_original_extent() -> None:
    from manga_translator.typesetter import plan_text_layout

    image = np.full((520, 320, 3), 245, dtype=np.uint8)
    local = np.zeros((360, 130), dtype=np.uint8)
    local[18:340, 12:55] = 255
    local[18:340, 76:119] = 255
    region = TextRegion(
        id="r0",
        x=90,
        y=70,
        w=130,
        h=360,
        source="ctd",
        font_size_hint=48,
        local_mask=local,
    )
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=True,
        ocr_text="あらゆる訓練を受け育った",
        mask=local,
    )
    cfg = TypesettingConfig(
        adaptive_bubble_layout=False,
        font_size_min=10,
        font_size_max=120,
        min_font_scale=0.85,
        font_preserve_floor_scale=0.92,
    )

    plan = plan_text_layout(
        image,
        group,
        {"r0": region},
        "接受各種訓練長大",
        PRIMARY,
        cfg,
        FALLBACK,
    )

    assert plan.fits
    assert plan.font_size >= 44
    assert plan.block_height >= 0.72 * 322
    assert max(len(chunk) for chunk in plan.chunks) - min(len(chunk) for chunk in plan.chunks) <= 1


def test_layout_rejects_instead_of_shrinking_to_tiny_text() -> None:
    from manga_translator.typesetter import plan_text_layout

    image = np.full((180, 140, 3), 255, dtype=np.uint8)
    local = np.zeros((90, 50), dtype=np.uint8)
    local[5:85, 8:42] = 255
    region = TextRegion(
        id="r0",
        x=40,
        y=40,
        w=50,
        h=90,
        source="ctd",
        font_size_hint=34,
        local_mask=local,
    )
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=True,
        ocr_text="短い文",
        mask=local,
    )
    cfg = TypesettingConfig(
        adaptive_bubble_layout=False,
        font_size_min=10,
        font_size_max=80,
        min_font_scale=0.85,
        font_preserve_floor_scale=0.92,
        reject_unreadable_layout=True,
    )

    plan = plan_text_layout(
        image,
        group,
        {"r0": region},
        "這是一段故意長到完全不可能在原本小對話框裡以可讀字級排下的翻譯文字" * 3,
        PRIMARY,
        cfg,
        FALLBACK,
    )

    assert not plan.fits
    assert plan.font_size >= 28
    assert plan.reason.startswith("translation_requires_font_below_")


def test_horizontal_layout_uses_tracking_to_fill_shorter_translation() -> None:
    from manga_translator.typesetter import plan_text_layout

    image = np.full((180, 420, 3), 255, dtype=np.uint8)
    local = np.zeros((70, 300), dtype=np.uint8)
    local[12:58, 12:288] = 255
    region = TextRegion(
        id="r0",
        x=50,
        y=50,
        w=300,
        h=70,
        source="ctd",
        font_size_hint=46,
        local_mask=local,
    )
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=False,
        ocr_text="ありがとうございました",
        mask=local,
    )
    cfg = TypesettingConfig(adaptive_bubble_layout=False, direction="horizontal")

    plan = plan_text_layout(
        image,
        group,
        {"r0": region},
        "非常感謝",
        PRIMARY,
        cfg,
        FALLBACK,
    )

    assert plan.fits
    assert plan.font_size >= 42
    assert plan.secondary_step > 0
    assert plan.block_width >= 0.55 * 276


def test_layout_plan_block_bbox_uses_global_coordinates() -> None:
    from manga_translator.typesetter import TextLayoutPlan, layout_plan_block_bbox

    plan = TextLayoutPlan(
        bbox=(100, 200, 300, 400),
        direction="vertical",
        font_size=40,
        chunks=("測試",),
        primary_step=0.0,
        secondary_step=45.0,
        center_x=80.0,
        center_y=120.0,
        block_width=40.0,
        block_height=90.0,
    )

    assert layout_plan_block_bbox(plan) == (160, 275, 40, 90)
