from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from manga_translator.config import InpaintingConfig
from manga_translator.stages.render import render_page_atomic
from manga_translator.typography.fonts import FontRole
from manga_translator.typography.layout import (
    AcceptedLayout,
    FontChoice,
    LayoutCandidate,
    LayoutDirection,
)
from manga_translator.typography.render import (
    AtomicRoiRequest,
    RenderStyle,
    build_atomic_roi_bbox,
    fit_render_style,
    render_layout_layer,
)
from manga_translator.typography.safe_region import SafeRegionArtifacts
from manga_translator.typography.shaping import ShapedFontRun
from manga_translator.typography.solver import layout_plan_hash


def _request(x: int = 80, y: int = 70, width: int = 90, height: int = 100) -> AtomicRoiRequest:
    render_mask = np.zeros((height, width), dtype=np.uint8)
    render_mask[8:-8, 8:-8] = 255
    alpha = np.zeros_like(render_mask)
    cv2.rectangle(alpha, (33, 25), (57, 75), 255, -1)
    inpaint_mask = np.zeros_like(render_mask)
    cv2.rectangle(inpaint_mask, (38, 30), (52, 70), 255, -1)
    safe = SafeRegionArtifacts(
        render_mask.copy(),
        render_mask,
        cv2.distanceTransform((render_mask > 0).astype(np.uint8), cv2.DIST_L2, 5),
        np.zeros_like(render_mask),
        0.95,
        "connected_background",
    )
    candidate = LayoutCandidate(
        FontChoice(FontRole.NEUTRAL_SANS),
        28,
        LayoutDirection.VERTICAL,
        ("中文",),
        (),
        1.0,
        0.0,
        (45.0, 50.0),
        12.0,
    )
    runs = (
        ShapedFontRun(
            text="銝剜?",
            font_sha256="a" * 64,
            font_path="fixture.ttf",
            glyph_coverage=(20013, 25991),
            direction="ttb",
            language="zh-Hant",
            features=("vert", "vrt2"),
            bbox=(33.0, 25.0, 57.0, 75.0),
            advance=50.0,
            anchor=(45.0, 50.0),
        ),
    )
    accepted = AcceptedLayout(
        candidate,
        alpha,
        1.0,
        0.0,
        layout_plan_hash(candidate, alpha, runs),
        runs,
    )
    return AtomicRoiRequest(
        (x, y, width, height),
        inpaint_mask,
        safe,
        accepted,
        RenderStyle(
            fill=(10, 20, 30, 255),
            stroke=(255, 255, 255, 255),
            stroke_width=1,
            shadow=(0, 0, 0, 100),
            shadow_offset=(2, 2),
        ),
    )


def test_atomic_render_exception_preserves_original_text_and_entire_page() -> None:
    original = np.full((260, 320, 3), 235, dtype=np.uint8)
    request = _request()
    x, y, width, height = request.roi_bbox
    original[y : y + height, x : x + width][request.inpaint_mask > 0] = 5

    def fail_renderer(_layout, _style):
        raise RuntimeError("simulated render failure")

    result = render_page_atomic(original, (request,), renderer=fail_renderer)

    assert not result.outcomes[0].committed
    assert result.outcomes[0].reason.startswith("rollback:RuntimeError")
    assert np.array_equal(result.image, original)


def test_roi_outside_pixels_are_byte_identical_after_success() -> None:
    original = np.full((260, 320, 3), 235, dtype=np.uint8)
    request = _request()
    x, y, width, height = request.roi_bbox
    original[y : y + height, x : x + width][request.inpaint_mask > 0] = 5
    result = render_page_atomic(
        original,
        (request,),
        InpaintingConfig(method="hybrid", mask_dilate=0, extra_mask_dilate=0),
    )

    outside = np.ones(original.shape[:2], dtype=bool)
    outside[y : y + height, x : x + width] = False
    assert result.outcomes[0].committed
    assert np.array_equal(result.image[outside], original[outside])
    assert not np.array_equal(result.image[y : y + height, x : x + width], original[y : y + height, x : x + width])


def test_post_render_containment_failure_rolls_back_inpaint() -> None:
    original = np.full((220, 260, 3), 245, dtype=np.uint8)
    request = _request(60, 50)
    x, y, width, height = request.roi_bbox
    original[y : y + height, x : x + width][request.inpaint_mask > 0] = 0

    def escaping_renderer(layout, style):
        layer = render_layout_layer(layout, style)
        layer[0:5, 0:5] = (255, 0, 0, 255)
        return layer

    result = render_page_atomic(original, (request,), renderer=escaping_renderer)

    assert not result.outcomes[0].committed
    assert "post-render alpha escaped" in result.outcomes[0].reason
    assert np.array_equal(result.image, original)


def test_empty_layout_or_render_alpha_never_commits_inpaint() -> None:
    original = np.full((220, 260, 3), 245, dtype=np.uint8)
    request = _request(60, 50)
    x, y, width, height = request.roi_bbox
    original[y : y + height, x : x + width][request.inpaint_mask > 0] = 0
    empty_layout = replace(
        request,
        layout=replace(request.layout, alpha=np.zeros_like(request.layout.alpha)),
    )

    preflight = render_page_atomic(original, (empty_layout,))
    assert not preflight.outcomes[0].committed
    assert preflight.outcomes[0].reason == "empty_layout_alpha"
    assert np.array_equal(preflight.image, original)

    def transparent_renderer(_layout, _style):
        return np.zeros((height, width, 4), dtype=np.uint8)

    postflight = render_page_atomic(original, (request,), renderer=transparent_renderer)
    assert not postflight.outcomes[0].committed
    assert "renderer returned empty alpha" in postflight.outcomes[0].reason
    assert np.array_equal(postflight.image, original)


def test_unshaped_or_tampered_layout_never_commits_inpaint() -> None:
    original = np.full((220, 260, 3), 245, dtype=np.uint8)
    request = _request(60, 50)

    unshaped = replace(request, layout=replace(request.layout, shaped_runs=()))
    missing = render_page_atomic(original, (unshaped,))
    assert missing.outcomes[0].reason == "missing_shaped_runs"
    assert np.array_equal(missing.image, original)

    tampered = replace(request, layout=replace(request.layout, plan_hash="tampered"))
    invalid = render_page_atomic(original, (tampered,))
    assert invalid.outcomes[0].reason == "invalid_layout_plan_hash"
    assert np.array_equal(invalid.image, original)


def test_empty_inpaint_mask_does_not_add_unpaired_translation() -> None:
    original = np.full((220, 260, 3), 245, dtype=np.uint8)
    request = _request(60, 50)
    request = replace(request, inpaint_mask=np.zeros_like(request.inpaint_mask))

    result = render_page_atomic(original, (request,))

    assert not result.outcomes[0].committed
    assert result.outcomes[0].reason == "empty_inpaint_mask"
    assert np.array_equal(result.image, original)


def test_high_contrast_source_text_residual_rolls_back_inpaint() -> None:
    original = np.full((220, 260, 3), 245, dtype=np.uint8)
    request = _request(60, 50)
    x, y, width, height = request.roi_bbox
    roi = original[y : y + height, x : x + width]
    roi[request.inpaint_mask > 0] = 0
    cv2.line(roi, (25, 28), (25, 70), (0, 0, 0), 5)
    guarded = replace(
        request,
        source_text_bbox=(15, 15, 55, 70),
        background_rgb=(245, 245, 245),
    )

    result = render_page_atomic(original, (guarded,))

    assert not result.outcomes[0].committed
    assert "source text remained" in result.outcomes[0].reason
    assert np.array_equal(result.image, original)


def test_single_working_page_profile_counts_roi_not_per_group_page_copies() -> None:
    original = np.full((1200, 1600, 3), 240, dtype=np.uint8)
    requests = (_request(100, 100), _request(500, 400), _request(900, 700))
    result = render_page_atomic(original, requests)

    assert result.profile.page_copies == 1
    assert result.profile.page_bytes_copied == original.nbytes
    assert result.profile.roi_bytes_copied == sum(
        request.roi_bbox[2] * request.roi_bbox[3] * original.shape[2] for request in requests
    )
    assert result.profile.roi_bytes_copied < original.nbytes // 20


def test_styled_layer_contains_fill_stroke_shadow_and_rotation_safe_roi_margin() -> None:
    request = _request()
    layer = render_layout_layer(request.layout, request.style)
    alpha = layer[:, :, 3]
    assert np.count_nonzero(alpha) > np.count_nonzero(request.layout.alpha)
    assert request.safe_region.accepts_alpha(alpha)

    bbox = build_atomic_roi_bbox(
        (400, 500),
        (100, 120, 30, 40),
        (95, 115, 50, 60),
        stroke_width=3,
        shadow_offset=(4, -2),
        rotation_degrees=30,
    )
    assert bbox[0] < 95 and bbox[1] < 115
    assert bbox[0] + bbox[2] > 145
    assert bbox[1] + bbox[3] > 175


def test_unsafe_style_decoration_is_dropped_before_atomic_render() -> None:
    request = _request()
    unsafe = replace(request.style, shadow_offset=(35, 35), stroke_width=12)

    fitted = fit_render_style(request.layout, request.safe_region, unsafe)
    layer = render_layout_layer(request.layout, fitted)

    assert fitted != unsafe
    assert fitted.shadow is None
    assert request.safe_region.accepts_alpha(layer[:, :, 3])
