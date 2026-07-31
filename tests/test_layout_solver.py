from __future__ import annotations

import cv2
import numpy as np
import pytest

from manga_translator.typography import solver as solver_module
from manga_translator.typography.fonts import FontRole
from manga_translator.typography.layout import (
    AcceptedLayout,
    FontChoice,
    LayoutCandidate,
    LayoutDirection,
    LayoutOverflow,
    LayoutRequest,
    RasterizedLayout,
)
from manga_translator.typography.safe_region import SafeRegionArtifacts
from manga_translator.typography.shaping import ShapedFontRun
from manga_translator.typography.solver import (
    _candidate_score,
    _candidate_specs,
    estimate_source_line_count,
    estimate_source_line_gap_em,
    solve_layout,
    source_line_gap_options,
)


class RectangleRasterizer:
    def __init__(self, *, coverage: bool = True, shaping: bool = True) -> None:
        self.coverage = coverage
        self.shaping = shaping

    def rasterize(self, candidate: LayoutCandidate, shape: tuple[int, int]) -> RasterizedLayout:
        lengths = [len(chunk) for chunk in candidate.chunks]
        if candidate.direction is LayoutDirection.HORIZONTAL:
            width = round(candidate.font_size * max(lengths) * 0.55)
            height = round(candidate.font_size * len(lengths) * candidate.line_gap_em)
        else:
            width = round(candidate.font_size * len(lengths) * candidate.line_gap_em)
            height = round(candidate.font_size * max(lengths))
        x1 = round(candidate.anchor[0] - width / 2)
        y1 = round(candidate.anchor[1] - height / 2)
        x2, y2 = x1 + width, y1 + height
        clipped = x1 < 0 or y1 < 0 or x2 > shape[1] or y2 > shape[0]
        alpha = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(
            alpha,
            (max(0, x1), max(0, y1)),
            (min(shape[1] - 1, x2 - 1), min(shape[0] - 1, y2 - 1)),
            255,
            -1,
        )
        return RasterizedLayout(alpha, self.shaping, self.coverage, clipped)


class ProvenanceRasterizer(RectangleRasterizer):
    def __init__(self, font_sha256: str) -> None:
        super().__init__()
        self.font_sha256 = font_sha256

    def rasterize(self, candidate: LayoutCandidate, shape: tuple[int, int]) -> RasterizedLayout:
        base = super().rasterize(candidate, shape)
        direction = "ttb" if candidate.direction is LayoutDirection.VERTICAL else "ltr"
        run = ShapedFontRun(
            text="".join(candidate.chunks),
            font_sha256=self.font_sha256,
            font_path="font.ttf",
            glyph_coverage=tuple(sorted({ord(char) for char in "".join(candidate.chunks)})),
            direction=direction,
            language="zh-Hant-TW",
            features=(),
            bbox=(0.0, 0.0, 10.0, 20.0),
            advance=20.0,
            anchor=candidate.anchor,
        )
        return RasterizedLayout(
            base.alpha,
            base.shaping_succeeded,
            base.glyph_coverage_complete,
            base.clipped,
            shaped_runs=(run,),
        )


def _safe_region(width: int = 180, height: int = 220, inset: int = 12) -> SafeRegionArtifacts:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[inset : height - inset, inset : width - inset] = 255
    signed = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return SafeRegionArtifacts(mask, mask.copy(), signed, np.zeros_like(mask), 0.9, "connected_background")


def _request(**overrides) -> LayoutRequest:
    values = {
        "text": "他說「你好」，2026年再見。",
        "safe_region": _safe_region(),
        "fonts": (FontChoice(FontRole.NEUTRAL_SANS, 400),),
        "source_font_size": 28.0,
        "source_center": (90.0, 110.0),
        "source_angle_degrees": 0.0,
        "hard_font_floor": 18,
        "max_lines": 4,
        "beam_width": 4096,
    }
    values.update(overrides)
    return LayoutRequest(**values)


def test_solver_is_deterministic_and_accepts_only_contained_alpha() -> None:
    request = _request()
    first = solve_layout(request, RectangleRasterizer())
    second = solve_layout(request, RectangleRasterizer())

    assert isinstance(first, AcceptedLayout)
    assert isinstance(second, AcceptedLayout)
    assert first.plan_hash == second.plan_hash
    assert first.candidate == second.candidate
    assert first.containment >= 0.995
    assert first.candidate.font_size >= request.hard_font_floor
    assert first.candidate.tracking_em <= 0.2


def test_solver_uses_legal_breaks_and_natural_spacing() -> None:
    result = solve_layout(_request(), RectangleRasterizer())

    assert isinstance(result, AcceptedLayout)
    assert result.candidate.line_gap_em <= 1.12
    assert result.candidate.tracking_em <= 0.2
    assert result.warnings == ()


def test_solver_retains_shaped_font_hash_provenance_in_plan_hash() -> None:
    first = solve_layout(_request(), ProvenanceRasterizer("a" * 64))
    second = solve_layout(_request(), ProvenanceRasterizer("b" * 64))

    assert isinstance(first, AcceptedLayout)
    assert isinstance(second, AcceptedLayout)
    assert first.shaped_runs[0].font_sha256 == "a" * 64
    assert second.shaped_runs[0].font_sha256 == "b" * 64
    assert first.plan_hash != second.plan_hash


def test_preferred_breaks_are_forwarded_to_candidate_scoring() -> None:
    request = _request(
        text="甲乙丙丁戊",
        max_lines=2,
        preferred_grapheme_breaks=(3,),
    )
    candidates = [candidate for _rough, candidate in _candidate_specs(request)]

    assert any(candidate.break_indices == (3,) for candidate in candidates)


def test_source_direction_and_line_count_are_bounded_by_source_evidence() -> None:
    request = _request(
        source_direction=LayoutDirection.VERTICAL,
        allow_alternate_direction=False,
        source_line_count=2,
        line_count_tolerance=1,
        max_lines=8,
        directions=(LayoutDirection.VERTICAL, LayoutDirection.HORIZONTAL),
    )
    candidates = [candidate for _rough, candidate in _candidate_specs(request)]

    assert {candidate.direction for candidate in candidates} == {LayoutDirection.VERTICAL}
    assert {len(candidate.chunks) for candidate in candidates} <= {1, 2, 3}
    assert {len(candidate.chunks) for candidate in candidates}


def test_font_search_never_goes_below_ninety_percent_of_reliable_source() -> None:
    request = _request(source_font_size=28.0, hard_font_floor=10)
    candidates = [candidate for _rough, candidate in _candidate_specs(request)]

    assert min(candidate.font_size for candidate in candidates) == 26


def test_candidate_score_uses_text_block_bbox_not_glyph_pixel_density() -> None:
    request = _request(source_text_bbox=(20, 30, 80, 120))
    candidate = next(candidate for _rough, candidate in _candidate_specs(request))
    solid = np.zeros((220, 180), dtype=np.uint8)
    sparse = np.zeros_like(solid)
    solid[30:150, 20:100] = 255
    cv2.rectangle(sparse, (20, 30), (99, 149), 255, 1)

    assert _candidate_score(request, candidate, solid) == _candidate_score(
        request, candidate, sparse
    )


def test_solver_reuses_line_break_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = solver_module.analyze_line_breaks

    def counted_analysis(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "analyze_line_breaks", counted_analysis)

    result = solve_layout(_request(), RectangleRasterizer())

    assert isinstance(result, AcceptedLayout)
    assert calls == 1


def test_source_line_count_uses_secondary_block_axis() -> None:
    mask = np.zeros((220, 180), dtype=np.uint8)
    mask[20:200, 30:120] = 255

    assert estimate_source_line_count(mask, LayoutDirection.VERTICAL, 28.0) == 3
    assert estimate_source_line_count(mask, LayoutDirection.HORIZONTAL, 28.0) == 6


def test_line_gap_search_stays_centered_on_source_spacing() -> None:
    estimated = estimate_source_line_gap_em(
        (20, 30, 180, 300),
        LayoutDirection.VERTICAL,
        60.0,
        2,
    )

    assert estimated == 2.0
    assert source_line_gap_options(estimated) == (1.8, 2.0, 2.2)


def test_missing_glyph_or_shaping_failure_is_a_hard_rejection() -> None:
    missing = solve_layout(_request(), RectangleRasterizer(coverage=False))
    failed = solve_layout(_request(), RectangleRasterizer(shaping=False))

    assert isinstance(missing, LayoutOverflow)
    assert missing.reason == "missing_glyph"
    assert isinstance(failed, LayoutOverflow)
    assert failed.reason == "shaping_failed"


def test_neighbor_collision_is_a_hard_rejection() -> None:
    neighbor = np.full((220, 180), 255, dtype=np.uint8)
    result = solve_layout(_request(neighbor_mask=neighbor), RectangleRasterizer())

    assert isinstance(result, LayoutOverflow)
    assert result.reason == "neighbor_collision"


def test_overflow_never_shrinks_below_floor_and_returns_actionable_capacity() -> None:
    result = solve_layout(
        _request(
            text="非常長的翻譯內容" * 20,
            safe_region=_safe_region(70, 70, 16),
            source_center=(35.0, 35.0),
            hard_font_floor=24,
            max_lines=2,
        ),
        RectangleRasterizer(),
    )

    assert isinstance(result, LayoutOverflow)
    assert result.available_size == (38, 38)
    assert result.grapheme_count == 160
    assert 0 <= result.suggested_max_graphemes < result.grapheme_count
    assert result.suggested_max_lines >= 1


def test_alpha_outside_safe_region_is_rejected_without_clipping_it_to_fit() -> None:
    result = solve_layout(
        _request(safe_region=_safe_region(90, 100, 40), source_center=(45.0, 50.0)),
        RectangleRasterizer(),
    )

    assert isinstance(result, LayoutOverflow)
    rejected = dict(result.rejected)
    assert rejected.get("alpha_outside_render_mask", 0) > 0 or rejected.get("clipped", 0) > 0
