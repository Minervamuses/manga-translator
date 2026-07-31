"""Deterministic beam search over shaped, raster-verified layout candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import product
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from ..text import grapheme_clusters
from .breaking import (
    analyze_line_breaks,
    balanced_legal_breaks,
    validate_breaks,
)
from .fonts import FontResolver, FontRole, MissingGlyphError
from .layout import (
    AcceptedLayout,
    LayoutCandidate,
    LayoutDirection,
    LayoutOverflow,
    LayoutRequest,
    LayoutResult,
    RasterizedLayout,
)
from .shaping import PillowRaqmEngine, RunShaper, ShapedFontRun


class CandidateRasterizer(Protocol):
    def rasterize(self, candidate: LayoutCandidate, shape: tuple[int, int]) -> RasterizedLayout: ...


def _chunks_for_breaks(text: str, breaks: tuple[int, ...]) -> tuple[str, ...]:
    chunks: list[str] = []
    start = 0
    for end in (*breaks, len(text)):
        chunk = text[start:end].removesuffix("\r\n").removesuffix("\n").removesuffix("\r")
        chunks.append(chunk)
        start = end
    return tuple(chunks)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return (0, 0, 0, 0)
    return cv2.boundingRect(points)


def text_block_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Return the rendered block extent, not the number of painted glyph pixels."""

    return _mask_bbox(alpha)


def estimate_source_line_count(
    source_mask: np.ndarray,
    direction: LayoutDirection,
    source_font_size: float,
) -> int:
    """Estimate source columns/rows from its block extent and reliable font scale."""

    _x, _y, width, height = _mask_bbox(source_mask)
    secondary_extent = width if direction is LayoutDirection.VERTICAL else height
    pitch = max(1.0, float(source_font_size) * 1.08)
    return max(1, round(secondary_extent / pitch))


def estimate_source_line_gap_em(
    source_bbox: tuple[int, int, int, int],
    direction: LayoutDirection,
    source_font_size: float,
    source_line_count: int,
) -> float:
    """Estimate the distance between source columns/rows in font-relative units."""

    if source_line_count <= 1:
        return 1.0
    secondary_extent = source_bbox[2] if direction is LayoutDirection.VERTICAL else source_bbox[3]
    gap = (secondary_extent - source_font_size) / max(1, source_line_count - 1)
    return max(0.80, min(2.50, gap / max(1.0, source_font_size)))


def source_line_gap_options(source_line_gap_em: float) -> tuple[float, ...]:
    """Search a small deterministic neighborhood around the source spacing."""

    return tuple(
        dict.fromkeys(
            round(max(0.75, min(2.75, source_line_gap_em * scale)), 3)
            for scale in (0.90, 1.0, 1.10)
        )
    )


def _candidate_directions(request: LayoutRequest) -> tuple[LayoutDirection, ...]:
    if request.source_direction is None:
        return tuple(dict.fromkeys(request.directions))
    if not request.allow_alternate_direction:
        return (request.source_direction,)
    return tuple(dict.fromkeys((request.source_direction, *request.directions)))


def _candidate_line_counts(request: LayoutRequest) -> range:
    if request.source_line_count is None:
        return range(1, request.max_lines + 1)
    tolerance = max(0, request.line_count_tolerance)
    lower = max(1, request.source_line_count - tolerance)
    upper = min(request.max_lines, request.source_line_count + tolerance)
    return range(lower, max(lower, upper) + 1)


def _block_geometry(
    bbox: tuple[int, int, int, int],
    direction: LayoutDirection,
) -> tuple[float, float, float, tuple[float, float]]:
    x, y, width, height = bbox
    primary = float(height if direction is LayoutDirection.VERTICAL else width)
    secondary = float(width if direction is LayoutDirection.VERTICAL else height)
    return primary, secondary, float(width * height), (x + width / 2.0, y + height / 2.0)


def _geometry_ratios(
    request: LayoutRequest,
    candidate: LayoutCandidate,
    alpha: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if request.source_text_bbox is None:
        return None
    candidate_bbox = _mask_bbox(alpha)
    if candidate_bbox[2] <= 0 or candidate_bbox[3] <= 0:
        return (0.0, 0.0, 0.0, math.inf)
    source_primary, source_secondary, source_area, source_center = _block_geometry(
        request.source_text_bbox,
        candidate.direction,
    )
    primary, secondary, area, center = _block_geometry(candidate_bbox, candidate.direction)
    center_offset_ratio = math.dist(center, source_center) / max(
        1.0,
        math.hypot(request.source_text_bbox[2], request.source_text_bbox[3]),
    )
    return (
        primary / max(1.0, source_primary),
        secondary / max(1.0, source_secondary),
        area / max(1.0, source_area),
        center_offset_ratio,
    )


def _line_break_preferences(request: LayoutRequest) -> dict[int, int]:
    return {
        item.index: item.preference
        for item in analyze_line_breaks(
            request.text,
            preferred_grapheme_breaks=request.preferred_grapheme_breaks,
        ).opportunities
    }


def _candidate_score(
    request: LayoutRequest,
    candidate: LayoutCandidate,
    alpha: np.ndarray,
    *,
    break_preferences: Mapping[int, int] | None = None,
) -> float:
    bbox = _mask_bbox(alpha)
    center = (
        (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        if bbox[2] > 0 and bbox[3] > 0
        else candidate.anchor
    )
    lengths = [len(grapheme_clusters(chunk)) for chunk in candidate.chunks]
    imbalance = (max(lengths) - min(lengths)) if lengths else 0
    preferences = (
        break_preferences
        if break_preferences is not None
        else _line_break_preferences(request)
    )
    semantic_reward = sum(preferences.get(index, 0) for index in candidate.break_indices)
    geometry = _geometry_ratios(request, candidate, alpha)
    geometry_cost = 0.0
    if geometry is not None:
        primary_ratio, secondary_ratio, area_ratio, center_offset_ratio = geometry
        geometry_cost = (
            0.8 * abs(math.log(max(primary_ratio, 1e-6)))
            + 0.8 * abs(math.log(max(secondary_ratio, 1e-6)))
            + 0.4 * abs(math.log(max(area_ratio, 1e-6)))
            + 2.0 * center_offset_ratio
        )
    direction_cost = (
        6.0
        if request.source_direction is not None
        and candidate.direction is not request.source_direction
        else 0.0
    )
    line_count_cost = (
        0.8 * abs(len(candidate.chunks) - request.source_line_count)
        if request.source_line_count is not None
        else 0.0
    )
    return (
        4.0 * abs(math.log(candidate.font_size / max(request.source_font_size, 1.0)))
        + 0.8
        * math.dist(center, request.source_center)
        / max(request.safe_region.render_mask.shape)
        + 0.55 * abs(candidate.rotation_degrees - request.source_angle_degrees) / 45.0
        + 0.35 * abs(candidate.font.weight - request.source_weight) / 500.0
        + 0.30
        * abs(candidate.line_gap_em - (request.source_line_gap_em or 1.0))
        + 1.4 * candidate.tracking_em
        + 0.08 * imbalance
        + geometry_cost
        + direction_cost
        + line_count_cost
        - 0.0005 * semantic_reward
    )


def layout_plan_hash(
    candidate: LayoutCandidate,
    alpha: np.ndarray,
    shaped_runs: tuple[ShapedFontRun, ...],
) -> str:
    """Fingerprint the exact shaped candidate and raster accepted for rendering."""

    payload = {
        "font": candidate.font.role.value,
        "weight": candidate.font.weight,
        "size": candidate.font_size,
        "direction": candidate.direction.value,
        "chunks": candidate.chunks,
        "breaks": candidate.break_indices,
        "line_gap_em": candidate.line_gap_em,
        "tracking_em": candidate.tracking_em,
        "anchor": candidate.anchor,
        "rotation": candidate.rotation_degrees,
        "alpha_sha256": hashlib.sha256(alpha.tobytes()).hexdigest(),
        "shaped_runs": [
            {
                "advance": run.advance,
                "anchor": run.anchor,
                "bbox": run.bbox,
                "direction": run.direction,
                "features": run.features,
                "font_sha256": run.font_sha256,
                "glyph_coverage": run.glyph_coverage,
                "language": run.language,
                "text": run.text,
            }
            for run in shaped_runs
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _anchors(request: LayoutRequest) -> tuple[tuple[float, float], ...]:
    x, y, width, height = _mask_bbox(request.safe_region.render_mask)
    mask_center = (x + width / 2.0, y + height / 2.0)
    return tuple(dict.fromkeys((request.source_center, mask_center)))


def _rotations(request: LayoutRequest) -> tuple[float, ...]:
    choices = request.rotation_options or (request.source_angle_degrees, 0.0)
    return tuple(dict.fromkeys(float(value) for value in choices))


def _candidate_specs(request: LayoutRequest) -> Iterable[tuple[float, LayoutCandidate]]:
    minimum_size = max(
        request.hard_font_floor,
        math.ceil(request.source_font_size * request.minimum_source_font_scale),
    )
    maximum_size = max(minimum_size, round(request.source_font_size * 1.15))
    sizes = range(maximum_size, minimum_size - 1, -1)
    for line_count in _candidate_line_counts(request):
        breaks = balanced_legal_breaks(
            request.text,
            line_count,
            preferred_grapheme_breaks=request.preferred_grapheme_breaks,
        )
        chunks = _chunks_for_breaks(request.text, breaks)
        if len(chunks) != line_count and line_count > 1:
            continue
        for font, size, direction, gap, tracking, anchor, rotation in product(
            request.fonts,
            sizes,
            _candidate_directions(request),
            request.line_gap_options,
            request.tracking_options,
            _anchors(request),
            _rotations(request),
        ):
            candidate = LayoutCandidate(
                font=font,
                font_size=size,
                direction=direction,
                chunks=chunks,
                break_indices=breaks,
                line_gap_em=gap,
                tracking_em=tracking,
                anchor=anchor,
                rotation_degrees=rotation,
            )
            rough = (
                abs(size - request.source_font_size) / max(request.source_font_size, 1.0)
                + 2.0 * tracking
                + abs(gap - (request.source_line_gap_em or 1.0))
                + 0.1
                * abs(line_count - (request.source_line_count or 2))
            )
            yield rough, candidate


def solve_layout(request: LayoutRequest, rasterizer: CandidateRasterizer) -> LayoutResult:
    """Return the best hard-valid candidate, or a structured overflow result."""

    if not request.text or not request.fonts:
        return _overflow(request, Counter({"empty_text_or_fonts": 1}))
    shape = request.safe_region.render_mask.shape
    if request.neighbor_mask is not None and request.neighbor_mask.shape != shape:
        raise ValueError("neighbor_mask must share the safe-region ROI-local shape")

    rejected: Counter[str] = Counter()
    break_preferences: dict[int, int] | None = None
    feasible: list[
        tuple[
            float,
            tuple[object, ...],
            LayoutCandidate,
            np.ndarray,
            float,
            tuple[ShapedFontRun, ...],
        ]
    ] = []
    specs = sorted(_candidate_specs(request), key=lambda item: (item[0], item[1].stable_key()))
    for _rough, candidate in specs[: request.beam_width]:
        reliable_font_floor = max(
            request.hard_font_floor,
            math.ceil(request.source_font_size * request.minimum_source_font_scale),
        )
        if candidate.font_size < reliable_font_floor:
            rejected["font_below_hard_floor"] += 1
            continue
        if validate_breaks(request.text, candidate.break_indices):
            rejected["clreq_violation"] += 1
            continue
        raster = rasterizer.rasterize(candidate, shape)
        if not raster.shaping_succeeded:
            rejected["shaping_failed"] += 1
            continue
        if not raster.glyph_coverage_complete:
            rejected["missing_glyph"] += 1
            continue
        if raster.alpha.shape != shape:
            rejected["invalid_alpha_shape"] += 1
            continue
        if raster.clipped:
            rejected["clipped"] += 1
            continue
        containment = request.safe_region.alpha_containment(raster.alpha)
        if containment < request.minimum_containment:
            rejected["alpha_outside_render_mask"] += 1
            continue
        if request.neighbor_mask is not None and np.any(
            (raster.alpha > 0) & (request.neighbor_mask > 0)
        ):
            rejected["neighbor_collision"] += 1
            continue
        geometry = _geometry_ratios(request, candidate, raster.alpha)
        if geometry is not None:
            primary_ratio, secondary_ratio, area_ratio, center_offset_ratio = geometry
            if primary_ratio < request.minimum_primary_axis_ratio:
                rejected["text_block_primary_axis_too_small"] += 1
                continue
            if secondary_ratio < request.minimum_secondary_axis_ratio:
                rejected["text_block_secondary_axis_too_small"] += 1
                continue
            if area_ratio < request.minimum_text_block_area_ratio:
                rejected["text_block_area_too_small"] += 1
                continue
            if center_offset_ratio > request.maximum_center_offset_ratio:
                rejected["text_block_off_center"] += 1
                continue
        if break_preferences is None:
            break_preferences = _line_break_preferences(request)
        score = _candidate_score(
            request,
            candidate,
            raster.alpha,
            break_preferences=break_preferences,
        )
        feasible.append(
            (
                score,
                candidate.stable_key(),
                candidate,
                raster.alpha,
                containment,
                raster.shaped_runs,
            )
        )

    normal_tracking = [item for item in feasible if item[2].tracking_em <= 0.2]
    if normal_tracking:
        feasible = normal_tracking
    elif feasible:
        rejected["excessive_tracking_only_solution"] += len(feasible)
        feasible = []
    if not feasible:
        return _overflow(request, rejected)
    score, _key, candidate, alpha, containment, shaped_runs = min(
        feasible, key=lambda item: (item[0], item[1])
    )
    warnings = ("tracking_exceeds_0.2em",) if candidate.tracking_em > 0.2 else ()
    return AcceptedLayout(
        candidate=candidate,
        alpha=alpha,
        containment=containment,
        score=score,
        plan_hash=layout_plan_hash(candidate, alpha, shaped_runs),
        shaped_runs=shaped_runs,
        warnings=warnings,
    )


def _overflow(request: LayoutRequest, rejected: Counter[str]) -> LayoutOverflow:
    _x, _y, width, height = _mask_bbox(request.safe_region.render_mask)
    cells = max(0, int((width * height) / max(1, request.hard_font_floor**2)))
    return LayoutOverflow(
        available_size=(width, height),
        grapheme_count=len(grapheme_clusters(request.text)),
        suggested_max_graphemes=cells,
        suggested_max_lines=max(1, min(request.max_lines, height // max(1, request.hard_font_floor))),
        reason=max(rejected, key=rejected.get) if rejected else "no_candidates",
        rejected=tuple(sorted(rejected.items())),
    )


class PillowLayoutRasterizer:
    """Rasterize full shaped runs; construction explicitly requires RAQM."""

    def __init__(self, resolver: FontResolver, *, stroke_width: int = 0) -> None:
        self.engine = PillowRaqmEngine()
        self.shaper = RunShaper(resolver, self.engine)
        self.stroke_width = stroke_width
        self._shape_cache: dict[
            tuple[str, FontRole, int, str, int], tuple[ShapedFontRun, ...]
        ] = {}

    def _shape(
        self,
        text: str,
        *,
        role: FontRole,
        size: int,
        direction: str,
    ) -> tuple[ShapedFontRun, ...]:
        key = (text, role, size, direction, self.stroke_width)
        cached = self._shape_cache.get(key)
        if cached is None:
            cached = self.shaper.shape(
                text,
                role=role,
                size=size,
                direction=direction,
                stroke_width=self.stroke_width,
            )
            self._shape_cache[key] = cached
        return cached

    def rasterize(self, candidate: LayoutCandidate, shape: tuple[int, int]) -> RasterizedLayout:
        if candidate.tracking_em != 0:
            return RasterizedLayout(
                np.zeros(shape, dtype=np.uint8),
                shaping_succeeded=False,
                glyph_coverage_complete=True,
                clipped=False,
                diagnostics=("Pillow RAQM does not expose synthetic tracking",),
            )
        canvas = Image.new("L", (shape[1], shape[0]), 0)
        direction = "ttb" if candidate.direction is LayoutDirection.VERTICAL else "ltr"
        try:
            shaped_lines = [
                self._shape(
                    chunk,
                    role=candidate.font.role,
                    size=candidate.font_size,
                    direction=direction,
                )
                for chunk in candidate.chunks
            ]
        except MissingGlyphError as error:
            return RasterizedLayout(
                np.zeros(shape, dtype=np.uint8),
                shaping_succeeded=True,
                glyph_coverage_complete=False,
                clipped=False,
                diagnostics=(str(error),),
            )
        except Exception as error:  # noqa: BLE001 - hard rejection retains diagnostic
            return RasterizedLayout(
                np.zeros(shape, dtype=np.uint8),
                shaping_succeeded=False,
                glyph_coverage_complete=False,
                clipped=False,
                diagnostics=(str(error),),
            )

        gap = candidate.font_size * candidate.line_gap_em
        extents = [sum(run.advance for run in runs) for runs in shaped_lines]
        clipped = False
        positioned_runs: list[ShapedFontRun] = []
        for line_index, runs in enumerate(shaped_lines):
            extent = extents[line_index]
            if candidate.direction is LayoutDirection.HORIZONTAL:
                base_x = candidate.anchor[0] - extent / 2.0
                base_y = candidate.anchor[1] + (line_index - (len(shaped_lines) - 1) / 2.0) * gap
            else:
                base_x = candidate.anchor[0] - (line_index - (len(shaped_lines) - 1) / 2.0) * gap
                base_y = candidate.anchor[1] - extent / 2.0
            for run in runs:
                positioned = replace(run, anchor=(base_x + run.anchor[0], base_y + run.anchor[1]))
                positioned_runs.append(positioned)
                x1, y1, x2, y2 = positioned.bbox
                clipped |= (
                    positioned.anchor[0] + x1 < 0
                    or positioned.anchor[1] + y1 < 0
                    or positioned.anchor[0] + x2 > shape[1]
                    or positioned.anchor[1] + y2 > shape[0]
                )
                self.engine.render(
                    canvas,
                    positioned,
                    size=candidate.font_size,
                    fill=255,
                    stroke_width=self.stroke_width,
                )
        alpha = np.asarray(canvas, dtype=np.uint8)
        if candidate.rotation_degrees:
            matrix = cv2.getRotationMatrix2D(candidate.anchor, candidate.rotation_degrees, 1.0)
            alpha = cv2.warpAffine(alpha, matrix, (shape[1], shape[0]))
        clipped |= bool(
            np.any(alpha[0, :])
            or np.any(alpha[-1, :])
            or np.any(alpha[:, 0])
            or np.any(alpha[:, -1])
        )
        return RasterizedLayout(
            alpha,
            True,
            True,
            clipped,
            shaped_runs=tuple(positioned_runs),
        )
