"""Robust style estimation from original pixels and detector text masks."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .models import RGB, Estimate, ExtractedStyle, ShadowEstimate


def _unknown() -> Estimate:
    return Estimate(value=None, confidence=0.0, sample_count=0, status="unknown")


def _rgb_estimate(bgr_pixels: np.ndarray, *, minimum_samples: int = 4) -> Estimate[RGB]:
    count = len(bgr_pixels)
    if count < minimum_samples:
        return Estimate(value=None, confidence=0.0, sample_count=count, status="unknown")
    pixels = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    median_lab = np.median(lab, axis=0)
    distances = np.linalg.norm(lab - median_lab, axis=1)
    keep = distances <= max(5.0, float(np.quantile(distances, 0.80)))
    robust = bgr_pixels[keep]
    median_bgr = np.median(robust, axis=0)
    dispersion = float(np.median(distances[keep])) if np.any(keep) else 100.0
    confidence = min(1.0, count / 48.0) * math.exp(-dispersion / 24.0)
    rgb = tuple(round(float(value)) for value in median_bgr[::-1])
    return Estimate(value=rgb, confidence=confidence, sample_count=count, status="known")


def _cluster_fill_and_stroke(
    crop: np.ndarray, mask: np.ndarray, distance: np.ndarray
) -> tuple[Estimate[RGB], Estimate[RGB]] | None:
    selected = mask > 0
    bgr = crop[selected]
    if len(bgr) < 16:
        return None
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(
        np.float32
    )
    centers = np.stack((np.median(lab, axis=0), lab[np.argmax(np.linalg.norm(lab - np.median(lab, axis=0), axis=1))]))
    labels = np.zeros(len(lab), dtype=np.int8)
    for _ in range(8):
        distances = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int8)
        updated = np.stack(
            [
                np.median(lab[labels == index], axis=0)
                if np.any(labels == index)
                else centers[index]
                for index in range(2)
            ]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    if float(np.linalg.norm(centers[0] - centers[1])) < 12.0:
        return None
    mask_distances = distance[selected]
    median_depth = [
        float(np.median(mask_distances[labels == index])) if np.any(labels == index) else 0.0
        for index in range(2)
    ]
    fill_index = int(np.argmax(median_depth))
    stroke_index = 1 - fill_index
    return _rgb_estimate(bgr[labels == fill_index]), _rgb_estimate(bgr[labels == stroke_index])


def _scalar(value: float | None, confidence: float, samples: int) -> Estimate[float]:
    if value is None or not math.isfinite(value) or samples <= 0:
        return Estimate(value=None, confidence=0.0, sample_count=max(0, samples), status="unknown")
    return Estimate(
        value=float(value),
        confidence=max(0.0, min(1.0, confidence)),
        sample_count=samples,
        status="known",
    )


def _color_distance(left: RGB, right: RGB) -> float:
    values = np.array([[left[::-1], right[::-1]]], dtype=np.uint8)
    lab = cv2.cvtColor(values, cv2.COLOR_BGR2LAB).astype(np.float32)[0]
    return float(np.linalg.norm(lab[0] - lab[1]))


def _shadow_estimate(crop: np.ndarray, mask: np.ndarray, fill: Estimate[RGB]) -> ShadowEstimate:
    kernel = np.ones((3, 3), np.uint8)
    background_ring = cv2.bitwise_not(cv2.dilate(mask, kernel, iterations=7))
    background = _rgb_estimate(crop[background_ring > 0], minimum_samples=8)
    if background.status == "unknown":
        return ShadowEstimate(
            color=_unknown(), offset=_unknown(), confidence=0.0, sample_count=0, status="unknown"
        )

    best: tuple[float, RGB, tuple[float, float], int] | None = None
    height, width = mask.shape
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if dx == 0 and dy == 0 or abs(dx) + abs(dy) < 2:
                continue
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
            candidate_mask = shifted & cv2.bitwise_not(cv2.dilate(mask, kernel, iterations=1))
            pixels = crop[candidate_mask > 0]
            estimate = _rgb_estimate(pixels, minimum_samples=8)
            if estimate.status == "unknown" or estimate.value is None or background.value is None:
                continue
            contrast = _color_distance(estimate.value, background.value)
            fill_distance = (
                _color_distance(estimate.value, fill.value)
                if fill.status == "known" and fill.value is not None
                else 0.0
            )
            pixel_lab = cv2.cvtColor(
                pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).reshape(-1, 3).astype(np.float32)
            reference_bgr = np.array(
                [[estimate.value[::-1], background.value[::-1]]], dtype=np.uint8
            )
            reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)
            supported = (np.linalg.norm(pixel_lab - reference_lab[0], axis=1) < 15.0) & (
                np.linalg.norm(pixel_lab - reference_lab[1], axis=1) > 15.0
            )
            support_count = int(np.count_nonzero(supported))
            coverage = support_count / max(1, int(np.count_nonzero(mask)))
            purity = support_count / max(1, len(pixels))
            score = (
                min(1.0, contrast / 35.0)
                * min(1.0, coverage / 0.28)
                * purity
                * estimate.confidence
            )
            if sum(estimate.value) / 3 > sum(background.value) / 3 - 10:
                score *= 0.2
            if fill_distance < 4.0:
                score *= 0.4
            if best is None or score > best[0]:
                best = (score, estimate.value, (float(dx), float(dy)), support_count)
    if best is None or best[0] < 0.58:
        return ShadowEstimate(
            color=_unknown(), offset=_unknown(), confidence=0.0, sample_count=0, status="unknown"
        )
    score, color, offset, samples = best
    return ShadowEstimate(
        color=Estimate(value=color, confidence=score, sample_count=samples, status="known"),
        offset=Estimate(value=offset, confidence=score, sample_count=samples, status="known"),
        confidence=score,
        sample_count=samples,
        status="known",
    )


def extract_style_fingerprint(
    original_image: np.ndarray,
    text_mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    source_angle: float = 0.0,
    source: str = "original_image",
) -> ExtractedStyle:
    """Extract style only from the immutable original image and aligned text mask."""

    if source != "original_image":
        raise ValueError("style extraction must use original_image")
    x, y, width, height = bbox
    crop = original_image[y : y + height, x : x + width]
    if crop.shape[:2] != (height, width):
        raise ValueError("bbox is outside original image")
    if text_mask.shape[:2] != (height, width):
        raise ValueError("text mask must be bbox-local and aligned")
    if crop.ndim != 3 or crop.shape[2] < 3:
        raise ValueError("original image must have at least three color channels")
    crop = crop[..., :3]
    mask = (text_mask > 0).astype(np.uint8) * 255
    ink_count = int(np.count_nonzero(mask))
    if ink_count == 0:
        unknown = _unknown()
        return ExtractedStyle(
            fill=unknown,
            stroke=unknown,
            stroke_width=unknown,
            background=unknown,
            ink_density=_scalar(0.0, 1.0, width * height),
            normalized_stroke_width=unknown,
            width_height_ratio=_scalar(width / max(height, 1), 1.0, 1),
            edge_roundness=unknown,
            stroke_variation=unknown,
            source_angle=_scalar(source_angle, 1.0, 1),
            shadow=ShadowEstimate(
                color=unknown, offset=unknown, confidence=0.0, sample_count=0, status="unknown"
            ),
        )

    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    positive = distance[distance > 0]
    boundary_limit = max(1.0, float(np.quantile(positive, 0.35)))
    core_mask = (distance > boundary_limit) & (mask > 0)
    boundary_mask = (distance > 0) & (distance <= boundary_limit)
    if np.count_nonzero(core_mask) < 4:
        core_mask = mask > 0
    clustered = _cluster_fill_and_stroke(crop, mask, distance)
    if clustered is None:
        fill = _rgb_estimate(crop[core_mask])
        stroke = Estimate(
            value=None,
            confidence=0.0,
            sample_count=int(np.count_nonzero(boundary_mask)),
            status="unknown",
        )
        stroke_width = _unknown()
    else:
        fill, stroke = clustered
        stroke_width = _scalar(boundary_limit, stroke.confidence, stroke.sample_count)

    background_ring = (cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=7) > 0) & (
        cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) == 0
    )
    background = _rgb_estimate(crop[background_ring], minimum_samples=12)

    skeleton_threshold = float(np.quantile(positive, 0.72))
    radii = positive[positive >= skeleton_threshold]
    glyph_width = float(2.0 * np.median(radii)) if len(radii) else None
    variation = (
        float(np.std(radii) / max(np.mean(radii), 1e-6)) if len(radii) > 1 else 0.0
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = float(sum(cv2.contourArea(contour) for contour in contours))
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    roundness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else None
    base_confidence = min(1.0, ink_count / 64.0)
    return ExtractedStyle(
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
        background=background,
        ink_density=_scalar(ink_count / (width * height), 1.0, width * height),
        normalized_stroke_width=_scalar(
            glyph_width / max(1, min(width, height)) if glyph_width is not None else None,
            base_confidence,
            len(radii),
        ),
        width_height_ratio=_scalar(width / height, 1.0, 1),
        edge_roundness=_scalar(roundness, base_confidence, len(contours)),
        stroke_variation=_scalar(variation, base_confidence, len(radii)),
        source_angle=_scalar(source_angle, 1.0, 1),
        shadow=_shadow_estimate(crop, mask, fill),
    )
