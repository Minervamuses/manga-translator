"""Conservative ROI-local render regions derived from image evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class SafeRegionArtifacts:
    """All masks use the same ROI-local coordinate system as ``roi``."""

    safe_mask: np.ndarray
    render_mask: np.ndarray
    signed_distance: np.ndarray
    protected_edges: np.ndarray
    confidence: float
    strategy: Literal["connected_background", "original_text_vicinity"]

    def alpha_containment(self, alpha: np.ndarray) -> float:
        if alpha.shape != self.render_mask.shape:
            raise ValueError("alpha and render_mask must share ROI-local shape")
        visible = alpha > 0
        count = int(np.count_nonzero(visible))
        if count == 0:
            return 1.0
        return float(np.count_nonzero(visible & (self.render_mask > 0)) / count)

    def accepts_alpha(self, alpha: np.ndarray, minimum: float = 0.995) -> bool:
        return self.alpha_containment(alpha) >= minimum


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask.shape[:2] != shape:
        raise ValueError(f"{name} must be ROI-local with shape {shape}, got {mask.shape[:2]}")
    return (mask > 0).astype(np.uint8)


def _polygon_mask(shape: tuple[int, int], polygons: tuple[Polygon, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    points = [np.rint(np.asarray(polygon)).astype(np.int32) for polygon in polygons if polygon]
    if points:
        cv2.fillPoly(result, points, 1)
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(mask, kernel, iterations=1)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.erode(mask, kernel, iterations=1)


def _edge_barrier(roi: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if roi.ndim == 2:
        gray = roi.astype(np.uint8, copy=False)
    elif roi.ndim == 3 and roi.shape[2] >= 3:
        gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("roi must be grayscale or BGR/RGB-compatible")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    nonzero = magnitude[magnitude > 0]
    adaptive_high = float(np.percentile(nonzero, 82)) if nonzero.size else 0.0
    gradient_edges = magnitude >= max(32.0, adaptive_high)
    canny = cv2.Canny(blurred, 24, 72) > 0
    raw_edges = gradient_edges | canny
    barrier = cv2.morphologyEx(
        raw_edges.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    barrier = _dilate(barrier, 1)
    barrier[_dilate(seed, 1) > 0] = 0
    return barrier, raw_edges.astype(np.uint8)


def _connected_to_seed(passable: np.ndarray, seed_area: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(passable, dtype=np.uint8)
    selected = np.unique(labels[seed_area > 0])
    selected = selected[selected != 0]
    if not selected.size:
        return np.zeros_like(passable, dtype=np.uint8)
    return np.isin(labels, selected).astype(np.uint8)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return (inside - outside).astype(np.float32)


def build_safe_region(
    roi: np.ndarray,
    original_text_mask: np.ndarray,
    *,
    line_polygons: tuple[Polygon, ...] = (),
    other_text_mask: np.ndarray | None = None,
    render_erosion: int = 2,
    low_confidence_threshold: float = 0.48,
) -> SafeRegionArtifacts:
    """Build usable and eroded render regions without assuming a bright background."""

    shape = roi.shape[:2]
    text = _binary_mask(original_text_mask, shape, "original_text_mask")
    if not np.any(text):
        raise ValueError("original_text_mask must contain real text pixels")
    polygon_seed = _polygon_mask(shape, line_polygons)
    seed = cv2.bitwise_or(text, polygon_seed)
    other = (
        np.zeros(shape, dtype=np.uint8)
        if other_text_mask is None
        else _binary_mask(other_text_mask, shape, "other_text_mask")
    )
    other = _dilate(other, 2)

    protected, raw_edges = _edge_barrier(roi, seed)
    protected[other > 0] = 1
    seed_ring = _dilate(seed, 5)
    seed_ring[_dilate(seed, 1) > 0] = 0
    seed_ring[protected > 0] = 0
    seed_ring[other > 0] = 0
    if not np.any(seed_ring):
        seed_ring = _dilate(seed, 2)
        seed_ring[protected > 0] = 0
        seed_ring[other > 0] = 0

    passable = (protected == 0) & (other == 0)
    safe = _connected_to_seed(passable, seed_ring)
    safe[seed > 0] = 1
    safe[other > 0] = 0
    safe[protected > 0] = 0

    evidence_ring = (_dilate(seed, 7) > 0) & (_dilate(seed, 1) == 0)
    evidence_pixels = int(np.count_nonzero(evidence_ring))
    edge_density = (
        float(np.count_nonzero(raw_edges[evidence_ring]) / evidence_pixels)
        if evidence_pixels
        else 1.0
    )
    support = (
        float(np.mean(safe[seed_ring > 0])) if np.any(seed_ring) else 0.0
    )
    if roi.ndim == 2:
        sample_image = roi[:, :, None]
    else:
        sample_image = roi[:, :, :3]
    samples = sample_image[evidence_ring]
    if samples.size:
        median = np.median(samples.astype(np.float32), axis=0)
        spread = float(np.mean(np.max(np.abs(samples.astype(np.float32) - median), axis=1)))
    else:
        spread = 255.0
    color_score = max(0.0, 1.0 - spread / 40.0)
    edge_score = max(0.0, 1.0 - edge_density * 5.0)
    confidence = float(np.clip(0.35 * support + 0.35 * color_score + 0.30 * edge_score, 0, 1))

    strategy: Literal["connected_background", "original_text_vicinity"]
    if confidence < low_confidence_threshold:
        safe = _dilate(seed, 3)
        safe[protected > 0] = 0
        safe[other > 0] = 0
        strategy = "original_text_vicinity"
    else:
        strategy = "connected_background"

    render = _erode(safe, max(0, render_erosion))
    render[protected > 0] = 0
    render[other > 0] = 0
    return SafeRegionArtifacts(
        safe_mask=(safe * 255).astype(np.uint8),
        render_mask=(render * 255).astype(np.uint8),
        signed_distance=_signed_distance(render),
        protected_edges=(protected * 255).astype(np.uint8),
        confidence=confidence,
        strategy=strategy,
    )
