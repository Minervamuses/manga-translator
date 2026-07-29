"""Styled run-layer composition and atomic ROI inpaint/render transactions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol

import cv2
import numpy as np

from ..config import InpaintingConfig
from ..inpainter import inpaint_roi
from ..style.models import ExtractedStyle
from .layout import AcceptedLayout
from .safe_region import SafeRegionArtifacts
from .solver import layout_plan_hash

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class RenderStyle:
    fill: RGBA = (0, 0, 0, 255)
    stroke: RGBA | None = None
    stroke_width: int = 0
    shadow: RGBA | None = None
    shadow_offset: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class AtomicRoiRequest:
    roi_bbox: tuple[int, int, int, int]
    inpaint_mask: np.ndarray = field(repr=False, compare=False)
    safe_region: SafeRegionArtifacts
    layout: AcceptedLayout
    style: RenderStyle = RenderStyle()
    source_text_bbox: tuple[int, int, int, int] | None = None
    background_rgb: RGB | None = None


@dataclass(frozen=True)
class AtomicRenderOutcome:
    committed: bool
    roi_bbox: tuple[int, int, int, int]
    changed_pixels: int
    roi_bytes_copied: int
    reason: str = ""


class LayerRenderer(Protocol):
    def __call__(self, layout: AcceptedLayout, style: RenderStyle) -> np.ndarray: ...


def _linear_channel(value: int) -> float:
    channel = max(0.0, min(1.0, value / 255.0))
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def color_contrast_ratio(left: RGB, right: RGB) -> float:
    """WCAG contrast ratio used as a conservative legibility guard."""

    def luminance(color: RGB) -> float:
        red, green, blue = (_linear_channel(value) for value in color)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    brighter, darker = sorted((luminance(left), luminance(right)), reverse=True)
    return (brighter + 0.05) / (darker + 0.05)


def has_high_contrast_text_residual(
    repaired: np.ndarray,
    *,
    source_text_bbox: tuple[int, int, int, int],
    background_rgb: RGB,
) -> bool:
    """Detect an interior text-like remnant after inpainting.

    The check is deliberately conservative: it only considers high-contrast
    connected components wholly inside the source text block. Components that
    touch the block boundary are ignored because they are commonly bubble or
    panel edges rather than missed glyph strokes.
    """

    x, y, width, height = source_text_bbox
    left, top = max(0, x), max(0, y)
    right = min(repaired.shape[1], x + width)
    bottom = min(repaired.shape[0], y + height)
    if right - left < 3 or bottom - top < 3:
        return False
    crop_rgb = repaired[top:bottom, left:right, :3][:, :, ::-1].astype(np.float32) / 255.0
    linear = np.where(
        crop_rgb <= 0.04045,
        crop_rgb / 12.92,
        ((crop_rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    background_luminance = (
        0.2126 * _linear_channel(background_rgb[0])
        + 0.7152 * _linear_channel(background_rgb[1])
        + 0.0722 * _linear_channel(background_rgb[2])
    )
    lighter = np.maximum(luminance, background_luminance)
    darker = np.minimum(luminance, background_luminance)
    contrast = (lighter + 0.05) / (darker + 0.05)
    high_contrast = (contrast >= 4.5).astype(np.uint8)
    high_contrast = cv2.morphologyEx(
        high_contrast,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(
        high_contrast,
        8,
    )
    minimum_area = max(8, round(width * height * 0.0004))
    crop_height, crop_width = high_contrast.shape
    for label in range(1, count):
        component_x = int(stats[label, cv2.CC_STAT_LEFT])
        component_y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        touches_boundary = (
            component_x <= 1
            or component_y <= 1
            or component_x + component_width >= crop_width - 1
            or component_y + component_height >= crop_height - 1
        )
        if component_area >= minimum_area and not touches_boundary:
            return True
    return False


def conservative_render_style(
    extracted: ExtractedStyle,
    *,
    maximum_stroke_width: int,
) -> RenderStyle:
    """Choose readable colors and keep decorations only with strong evidence."""

    background = extracted.background
    background_rgb = (
        background.value
        if background is not None
        and background.status == "known"
        and background.value is not None
        and background.confidence >= 0.65
        else None
    )
    background_lightness = (
        sum(background_rgb) / 3.0 if background_rgb is not None else None
    )
    if background_lightness is not None and background_lightness >= 190.0:
        return RenderStyle(fill=(0, 0, 0, 255))

    fallback_fill: RGB = (
        (255, 255, 255)
        if background_lightness is not None and background_lightness <= 80.0
        else (0, 0, 0)
    )
    values = extracted.renderer_values(
        min_confidence=0.80,
        default_fill=fallback_fill,
        default_stroke=None,
        stroke_min_confidence=0.80,
        minimum_stroke_contrast=96.0,
    )
    fill = tuple(int(value) for value in values["fill_rgb"])
    if background_rgb is not None and color_contrast_ratio(fill, background_rgb) < 4.5:
        fill = fallback_fill

    # Unknown/mid-tone backgrounds are intentionally undecorated.  A dark,
    # confidently sampled background may retain a high-confidence outline/shadow.
    allow_decoration = background_lightness is not None and background_lightness <= 80.0
    stroke_value = values["stroke_rgb"] if allow_decoration else None
    stroke = (
        (*tuple(int(value) for value in stroke_value), 255)
        if stroke_value is not None
        else None
    )
    shadow_value = values["shadow"] if allow_decoration else None
    shadow = (
        (*tuple(int(value) for value in shadow_value.color.value), 128)
        if shadow_value is not None
        and shadow_value.confidence >= 0.85
        and shadow_value.color.value is not None
        else None
    )
    shadow_offset = (
        tuple(round(value) for value in shadow_value.offset.value)
        if shadow is not None and shadow_value.offset.value is not None
        else (0, 0)
    )
    return RenderStyle(
        fill=(*fill, 255),
        stroke=stroke,
        stroke_width=(
            min(
                max(0, round(float(values["stroke_width"]))),
                max(0, maximum_stroke_width),
            )
            if stroke is not None
            else 0
        ),
        shadow=shadow,
        shadow_offset=shadow_offset,
    )


def build_atomic_roi_bbox(
    page_shape: tuple[int, int],
    inpaint_bbox: tuple[int, int, int, int],
    render_bbox: tuple[int, int, int, int],
    *,
    stroke_width: int = 0,
    shadow_offset: tuple[int, int] = (0, 0),
    rotation_degrees: float = 0.0,
) -> tuple[int, int, int, int]:
    """Union inpaint/render extents with stroke, shadow, and rotation margin."""

    height, width = page_shape
    ix, iy, iw, ih = inpaint_bbox
    rx, ry, rw, rh = render_bbox
    x1, y1 = min(ix, rx), min(iy, ry)
    x2, y2 = max(ix + iw, rx + rw), max(iy + ih, ry + rh)
    diagonal = math.hypot(rw, rh)
    rotation_margin = math.ceil(max(0.0, diagonal - min(rw, rh)) / 2.0) if rotation_degrees else 0
    shadow_margin = max(abs(shadow_offset[0]), abs(shadow_offset[1]))
    margin = max(2, stroke_width + 1, shadow_margin + 1, rotation_margin)
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _solid_layer(alpha: np.ndarray, color: RGBA) -> np.ndarray:
    layer = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    layer[:, :, :3] = color[:3]
    layer[:, :, 3] = (alpha.astype(np.uint16) * color[3] // 255).astype(np.uint8)
    return layer


def _over_rgba(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    bg = background.astype(np.float32) / 255.0
    fg = foreground.astype(np.float32) / 255.0
    fg_a = fg[:, :, 3:4]
    bg_a = bg[:, :, 3:4]
    out_a = fg_a + bg_a * (1.0 - fg_a)
    premultiplied = fg[:, :, :3] * fg_a + bg[:, :, :3] * bg_a * (1.0 - fg_a)
    out_rgb = np.divide(
        premultiplied,
        out_a,
        out=np.zeros_like(premultiplied),
        where=out_a > 0,
    )
    return np.clip(np.dstack((out_rgb, out_a)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_layout_layer(layout: AcceptedLayout, style: RenderStyle) -> np.ndarray:
    """Colorize the solver's shaped/rotated alpha with shadow, stroke, and fill."""

    alpha = layout.alpha.astype(np.uint8, copy=False)
    layer = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    if style.shadow is not None:
        dx, dy = style.shadow_offset
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shadow_alpha = cv2.warpAffine(alpha, matrix, (alpha.shape[1], alpha.shape[0]))
        layer = _over_rgba(layer, _solid_layer(shadow_alpha, style.shadow))
    if style.stroke is not None and style.stroke_width > 0:
        radius = style.stroke_width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        stroke_alpha = cv2.dilate(alpha, kernel, iterations=1)
        layer = _over_rgba(layer, _solid_layer(stroke_alpha, style.stroke))
    return _over_rgba(layer, _solid_layer(alpha, style.fill))


def fit_render_style(
    layout: AcceptedLayout,
    safe_region: SafeRegionArtifacts,
    requested: RenderStyle,
) -> RenderStyle:
    """Drop unsafe decoration while retaining the extracted fill color."""

    candidates = (
        requested,
        replace(requested, shadow=None, shadow_offset=(0, 0)),
        replace(
            requested,
            stroke=None,
            stroke_width=0,
            shadow=None,
            shadow_offset=(0, 0),
        ),
    )
    for candidate in candidates:
        layer = render_layout_layer(layout, candidate)
        if safe_region.accepts_alpha(layer[:, :, 3]):
            return candidate
    raise ValueError("accepted layout fill escaped its safe region")


def _composite_bgr(background: np.ndarray, layer: np.ndarray) -> np.ndarray:
    alpha = layer[:, :, 3:4].astype(np.uint16)
    foreground = layer[:, :, :3][:, :, ::-1].astype(np.uint16)
    base = background.astype(np.uint16)
    return ((foreground * alpha + base * (255 - alpha) + 127) // 255).astype(np.uint8)


def atomic_inpaint_render(
    working_image: np.ndarray,
    request: AtomicRoiRequest,
    inpainting: InpaintingConfig | None = None,
    *,
    renderer: LayerRenderer = render_layout_layer,
) -> AtomicRenderOutcome:
    """Commit a complete ROI only after inpaint, render, and validation succeed."""

    x, y, width, height = request.roi_bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "invalid_roi")
    if x + width > working_image.shape[1] or y + height > working_image.shape[0]:
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "roi_out_of_bounds")
    expected_shape = (height, width)
    artifacts = request.safe_region
    if (
        request.inpaint_mask.shape[:2] != expected_shape
        or artifacts.render_mask.shape != expected_shape
        or request.layout.alpha.shape != expected_shape
    ):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "roi_shape_mismatch")
    if not np.any(request.inpaint_mask):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "empty_inpaint_mask")
    if not np.any(request.layout.alpha):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "empty_layout_alpha")
    if not request.layout.shaped_runs:
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "missing_shaped_runs")
    if request.layout.plan_hash != layout_plan_hash(
        request.layout.candidate,
        request.layout.alpha,
        request.layout.shaped_runs,
    ):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "invalid_layout_plan_hash")
    if not artifacts.accepts_alpha(request.layout.alpha):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "pre_render_containment")

    original_roi = working_image[y : y + height, x : x + width].copy()
    try:
        repaired = inpaint_roi(original_roi, request.inpaint_mask, inpainting)
        if (
            request.source_text_bbox is not None
            and request.background_rgb is not None
            and has_high_contrast_text_residual(
                repaired,
                source_text_bbox=request.source_text_bbox,
                background_rgb=request.background_rgb,
            )
        ):
            raise ValueError("source text remained after inpainting")
        layer = renderer(request.layout, request.style)
        if layer.shape != (height, width, 4):
            raise ValueError("renderer returned non-ROI RGBA layer")
        if not np.any(layer[:, :, 3]):
            raise ValueError("renderer returned empty alpha")
        if not artifacts.accepts_alpha(layer[:, :, 3]):
            raise ValueError("post-render alpha escaped render mask")
        composed = _composite_bgr(repaired, layer)
    except Exception as error:  # noqa: BLE001 - transaction converts errors to rollback outcomes
        return AtomicRenderOutcome(
            False,
            request.roi_bbox,
            0,
            int(original_roi.nbytes),
            f"rollback:{type(error).__name__}:{error}",
        )

    changed = int(np.count_nonzero(np.any(composed != original_roi, axis=2)))
    working_image[y : y + height, x : x + width] = composed
    return AtomicRenderOutcome(True, request.roi_bbox, changed, int(original_roi.nbytes))
