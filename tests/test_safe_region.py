from __future__ import annotations

import cv2
import numpy as np
import pytest

from manga_translator.typography import safe_region as safe_region_module
from manga_translator.typography.safe_region import (
    build_safe_region,
    decode_safe_region_artifacts,
    encode_safe_region_artifacts,
)


def _shape_fixture(kind: str, *, dark: bool = False, gradient: bool = False):
    height, width = 140, 180
    outside = 185 if dark else 45
    inside = 34 if dark else 230
    image = np.full((height, width, 3), outside, dtype=np.uint8)
    ground_truth = np.zeros((height, width), dtype=np.uint8)
    if kind == "ellipse":
        cv2.ellipse(ground_truth, (90, 70), (66, 48), 0, 0, 360, 255, -1)
    else:
        points = np.array([[18, 24], [155, 20], [169, 70], [145, 119], [84, 126], [31, 111]])
        cv2.fillPoly(ground_truth, [points], 255)
    image[ground_truth > 0] = inside
    if gradient:
        ramp = np.linspace(-24, 24, width, dtype=np.float32)
        graded = np.clip(inside + ramp, 0, 255).astype(np.uint8)
        for x in range(width):
            image[:, x][ground_truth[:, x] > 0] = graded[x]
    contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (5, 5, 5), 3)
    text = np.zeros((height, width), dtype=np.uint8)
    text[60:80, 82:98] = 255
    return image, text, ground_truth


@pytest.mark.parametrize(
    ("kind", "dark", "gradient"),
    [
        ("ellipse", False, False),
        ("spike", False, False),
        ("ellipse", True, False),
        ("ellipse", False, True),
        ("spike", True, True),
    ],
)
def test_safe_region_does_not_cross_protected_shape_edges(
    kind: str,
    dark: bool,
    gradient: bool,
) -> None:
    image, text, expected = _shape_fixture(kind, dark=dark, gradient=gradient)
    artifacts = build_safe_region(image, text, render_erosion=1)

    escaped = np.count_nonzero((artifacts.safe_mask > 0) & (expected == 0))
    assert escaped <= 4
    assert np.count_nonzero(artifacts.render_mask) > np.count_nonzero(text)
    assert artifacts.signed_distance.shape == text.shape
    assert np.all(artifacts.signed_distance[artifacts.render_mask > 0] >= 0)


def test_other_text_group_is_excluded_from_safe_and_render_masks() -> None:
    image, text, _expected = _shape_fixture("ellipse")
    other = np.zeros_like(text)
    other[52:88, 125:143] = 255

    artifacts = build_safe_region(image, text, other_text_mask=other)

    assert not np.any(artifacts.safe_mask[other > 0])
    assert not np.any(artifacts.render_mask[other > 0])


def test_text_over_art_has_low_confidence_and_never_expands_aggressively() -> None:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    for offset in range(-120, 180, 8):
        cv2.line(image, (max(0, offset), max(0, -offset)), (min(159, offset + 119), min(119, 119)), (5, 5, 5), 2)
    text = np.zeros((120, 160), dtype=np.uint8)
    text[53:67, 73:87] = 255

    artifacts = build_safe_region(image, text)
    vicinity = cv2.dilate(text, np.ones((7, 7), dtype=np.uint8))

    assert artifacts.confidence < 0.48
    assert artifacts.strategy == "original_text_vicinity"
    assert not np.any((artifacts.safe_mask > 0) & (vicinity == 0))


def test_aligned_line_geometry_can_promote_caption_confidence() -> None:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    for offset in range(-120, 180, 8):
        cv2.line(
            image,
            (max(0, offset), max(0, -offset)),
            (min(159, offset + 119), min(119, 119)),
            (5, 5, 5),
            2,
        )
    text = np.zeros((120, 160), dtype=np.uint8)
    text[53:67, 73:87] = 255
    polygon = (((65.0, 45.0), (95.0, 45.0), (95.0, 75.0), (65.0, 75.0)),)

    without_geometry = build_safe_region(image, text)
    with_geometry = build_safe_region(image, text, line_polygons=polygon)

    assert without_geometry.confidence < 0.48
    assert with_geometry.confidence > without_geometry.confidence
    assert with_geometry.confidence >= 0.48
    assert with_geometry.strategy == "connected_background"


def test_alpha_containment_rejects_pixels_outside_eroded_render_mask() -> None:
    image, text, _expected = _shape_fixture("ellipse")
    artifacts = build_safe_region(image, text)
    alpha = artifacts.render_mask.copy()
    assert artifacts.accepts_alpha(alpha)

    alpha[0, 0] = 255
    assert not artifacts.accepts_alpha(alpha, minimum=1.0)


def test_render_erosion_preserves_detector_backed_text_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((80, 100, 3), 230, dtype=np.uint8)
    text = np.zeros((80, 100), dtype=np.uint8)
    text[18:63:4, 48] = 255

    def dense_edge_barrier(
        _roi: np.ndarray,
        seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        protected = np.ones_like(seed, dtype=np.uint8)
        protected[cv2.dilate(seed, np.ones((3, 3), dtype=np.uint8)) > 0] = 0
        return protected, np.zeros_like(seed, dtype=np.uint8)

    monkeypatch.setattr(safe_region_module, "_edge_barrier", dense_edge_barrier)

    artifacts = build_safe_region(
        image,
        text,
        render_erosion=2,
    )

    expected_clearance = cv2.dilate(
        text,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    assert artifacts.strategy == "connected_background"
    assert np.all(artifacts.render_mask[text > 0] == 255)
    assert np.all(artifacts.render_mask[expected_clearance > 0] == 255)


def test_masks_must_be_roi_local_and_original_mask_must_be_real() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ROI-local"):
        build_safe_region(image, np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="real text pixels"):
        build_safe_region(image, np.zeros((20, 30), dtype=np.uint8))


def test_safe_region_bundle_round_trip_is_lossless_and_deterministic() -> None:
    image, text, _expected = _shape_fixture("ellipse", gradient=True)
    artifacts = build_safe_region(image, text)

    first = encode_safe_region_artifacts(artifacts)
    second = encode_safe_region_artifacts(artifacts)
    decoded = decode_safe_region_artifacts(first)

    assert first == second
    assert decoded.confidence == artifacts.confidence
    assert decoded.strategy == artifacts.strategy
    for name in ("safe_mask", "render_mask", "signed_distance", "protected_edges"):
        assert np.array_equal(getattr(decoded, name), getattr(artifacts, name))

    with pytest.raises(ValueError, match="payload length"):
        decode_safe_region_artifacts(first + b"unexpected")
