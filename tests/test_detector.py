from __future__ import annotations

import numpy as np

from manga_translator.config import DetectionConfig, PostprocessConfig
from manga_translator.detector import (
    TextRegion,
    _conservative_text_mask,
    _extract_mask_fallback_regions,
    postprocess_regions,
)


def test_group_masks_are_local_and_preserve_pixel_mask() -> None:
    refined = np.zeros((100, 120), dtype=np.uint8)
    refined[20:30, 40:50] = 255
    region = TextRegion(id="", x=38, y=18, w=16, h=16, vertical=False)

    regions, groups = postprocess_regions(
        [region],
        (100, 120),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
        refined_mask=refined,
    )

    assert len(regions) == 1
    assert len(groups) == 1
    group = groups[0]
    assert group.mask is not None
    assert group.mask.shape == (group.h, group.w)
    assert int(np.count_nonzero(group.mask)) == 100


def test_mask_fallback_recovers_unboxed_text_like_cluster() -> None:
    mask = np.zeros((180, 180), dtype=np.uint8)
    # 模擬數個相鄰直排字的 segmentation pixels。
    for y in (35, 55, 75, 95):
        mask[y : y + 10, 92:100] = 255
        mask[y + 3 : y + 7, 88:104] = 255

    cfg = DetectionConfig(
        device="cpu",
        mask_fallback_enabled=True,
        mask_fallback_threshold=20,
        mask_fallback_min_area=12,
        mask_fallback_padding=3,
    )
    candidates = _extract_mask_fallback_regions(mask, [], cfg)

    assert candidates
    assert any(candidate.source == "mask_fallback" for candidate in candidates)
    assert any(candidate.vertical for candidate in candidates)
    assert all(candidate.local_mask is not None for candidate in candidates)
    assert all(np.any(candidate.local_mask) for candidate in candidates)


def test_mask_fallback_does_not_duplicate_covered_detector_box() -> None:
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[30:70, 50:65] = 255
    existing = [TextRegion(id="", x=45, y=25, w=28, h=55)]
    cfg = DetectionConfig(
        device="cpu",
        mask_fallback_enabled=True,
        mask_fallback_threshold=20,
        mask_fallback_min_area=8,
        mask_fallback_padding=4,
    )

    assert _extract_mask_fallback_regions(mask, existing, cfg) == []


def test_empty_refined_mask_does_not_fall_back_to_region_rectangle() -> None:
    refined = np.zeros((60, 80), dtype=np.uint8)
    region = TextRegion(id="", x=10, y=12, w=20, h=16)

    _regions, groups = postprocess_regions(
        [region],
        (60, 80),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
        refined_mask=refined,
    )

    assert groups[0].mask is not None
    assert int(np.count_nonzero(groups[0].mask)) == 0


def test_conservative_mask_rejects_refined_pixels_without_raw_support() -> None:
    raw = np.zeros((40, 50), dtype=np.uint8)
    refined = np.zeros_like(raw)
    raw[10:14, 12:16] = 255
    refined[8:18, 10:20] = 255
    refined[25:35, 30:40] = 255  # 模擬人物線稿被 refinement 誤納入

    safe = _conservative_text_mask(
        refined,
        raw,
        DetectionConfig(
            device="cpu",
            raw_support_threshold=30,
            raw_support_dilate=1,
        ),
    )

    assert np.any(safe[9:17, 11:17])
    assert not np.any(safe[25:35, 30:40])


def test_huge_container_does_not_group_separate_real_regions() -> None:
    huge = TextRegion(id="", x=0, y=0, w=1000, h=1000, vertical=False)
    first = TextRegion(id="", x=40, y=50, w=80, h=120, vertical=True)
    second = TextRegion(id="", x=700, y=700, w=80, h=120, vertical=True)

    _regions, groups = postprocess_regions(
        [huge, first, second],
        (1000, 1000),
        PostprocessConfig(min_region_area=1, enable_grouping=True),
        refined_mask=None,
    )

    assert len(groups) == 3
