"""Styled run-layer composition and atomic ROI inpaint/render transactions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ..config import InpaintingConfig
from ..inpainter import inpaint_roi
from .layout import AcceptedLayout
from .safe_region import SafeRegionArtifacts

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


@dataclass(frozen=True)
class AtomicRenderOutcome:
    committed: bool
    roi_bbox: tuple[int, int, int, int]
    changed_pixels: int
    roi_bytes_copied: int
    reason: str = ""


class LayerRenderer(Protocol):
    def __call__(self, layout: AcceptedLayout, style: RenderStyle) -> np.ndarray: ...


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
    if not artifacts.accepts_alpha(request.layout.alpha):
        return AtomicRenderOutcome(False, request.roi_bbox, 0, 0, "pre_render_containment")

    original_roi = working_image[y : y + height, x : x + width].copy()
    try:
        repaired = inpaint_roi(original_roi, request.inpaint_mask, inpainting)
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
