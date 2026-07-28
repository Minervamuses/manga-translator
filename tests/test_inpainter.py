from __future__ import annotations

import numpy as np

from manga_translator.config import InpaintingConfig
from manga_translator.detector import DetectionResult, TextGroup, TextRegion
from manga_translator.inpainter import inpaint_regions


def test_only_successfully_translated_local_mask_is_erased() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[:] = (80, 80, 80)

    good_mask = np.zeros((20, 20), dtype=np.uint8)
    good_mask[5:15, 5:15] = 255
    bad_mask = np.full((20, 20), 255, dtype=np.uint8)
    good = TextGroup(
        id="good",
        region_ids=[],
        bbox=(10, 10, 20, 20),
        vertical=False,
        translation="成功",
        translation_valid=True,
        mask=good_mask,
    )
    bad = TextGroup(
        id="bad",
        region_ids=[],
        bbox=(60, 10, 20, 20),
        vertical=False,
        translation="",
        translation_valid=False,
        mask=bad_mask,
    )
    detection = DetectionResult([], [], [good, bad], np.zeros((60, 100), dtype=np.uint8))
    cfg = InpaintingConfig(
        method="white",
        mask_dilate=0,
        extra_mask_dilate=0,
        only_translated_groups=True,
    )

    result = inpaint_regions(image, detection, cfg)

    assert np.all(result[15:25, 15:25] == 255)
    assert np.all(result[10:30, 60:80] == 80)


def test_zero_group_mask_preserves_original_by_default() -> None:
    image = np.full((40, 60, 3), 50, dtype=np.uint8)
    region = TextRegion(id="r0", x=10, y=8, w=15, h=12, local_mask=np.zeros((12, 15), np.uint8))
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=False,
        translation="成功",
        translation_valid=True,
        mask=np.zeros((12, 15), dtype=np.uint8),
    )
    detection = DetectionResult(
        [region],
        [region],
        [group],
        np.zeros((40, 60), dtype=np.uint8),
    )

    result = inpaint_regions(
        image,
        detection,
        InpaintingConfig(method="white", mask_dilate=0, extra_mask_dilate=0),
    )

    assert np.all(result[8:20, 10:25] == 50)


def test_bbox_fallback_is_available_only_when_explicitly_enabled() -> None:
    image = np.full((40, 60, 3), 50, dtype=np.uint8)
    region = TextRegion(id="r0", x=10, y=8, w=15, h=12)
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=False,
        translation="成功",
        translation_valid=True,
    )
    detection = DetectionResult([region], [region], [group], np.zeros((40, 60), np.uint8))

    result = inpaint_regions(
        image,
        detection,
        InpaintingConfig(
            method="white",
            mask_dilate=0,
            extra_mask_dilate=0,
            allow_bbox_fallback=True,
        ),
    )

    assert np.all(result[8:20, 10:25] == 255)


def test_flat_background_refinement_removes_antialiased_text_edges_only() -> None:
    image = np.full((70, 110, 3), 245, dtype=np.uint8)
    # Simulate a glyph whose detector mask contains only the black core while
    # the original image still has a wider gray antialiasing halo.
    image[20:50, 30:46] = 120
    image[23:47, 33:43] = 20
    # Nearby panel line must survive; it is outside the small edge-search radius.
    image[10:60, 55:58] = 0

    core = np.zeros((40, 30), dtype=np.uint8)
    core[8:32, 8:18] = 255
    group = TextGroup(
        id="g0",
        region_ids=[],
        bbox=(25, 15, 30, 40),
        vertical=True,
        translation="測試",
        translation_valid=True,
        mask=core,
    )
    detection = DetectionResult([], [], [group], np.zeros((70, 110), dtype=np.uint8))
    cfg = InpaintingConfig(
        method="hybrid",
        mask_dilate=1,
        extra_mask_dilate=0,
        hybrid_flat_edge_expand=3,
        hybrid_flat_edge_contrast=8,
        hybrid_flat_edge_max_growth=5.0,
    )

    result = inpaint_regions(image, detection, cfg)

    repaired_text = result[20:50, 30:46]
    assert float(np.mean(repaired_text[:, :, 0] < 230)) < 0.06
    assert np.all(result[10:60, 55:58] == 0)
