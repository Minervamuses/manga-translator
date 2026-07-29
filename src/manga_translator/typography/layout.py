"""Typed contracts shared by the deterministic layout solver and renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .fonts import FontRole
from .safe_region import SafeRegionArtifacts
from .shaping import ShapedFontRun


class LayoutDirection(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True)
class FontChoice:
    role: FontRole
    weight: int = 400


@dataclass(frozen=True)
class LayoutCandidate:
    font: FontChoice
    font_size: int
    direction: LayoutDirection
    chunks: tuple[str, ...]
    break_indices: tuple[int, ...]
    line_gap_em: float
    tracking_em: float
    anchor: tuple[float, float]
    rotation_degrees: float

    def stable_key(self) -> tuple[object, ...]:
        return (
            self.font.role.value,
            self.font.weight,
            -self.font_size,
            self.direction.value,
            self.chunks,
            self.break_indices,
            self.line_gap_em,
            self.tracking_em,
            self.anchor,
            self.rotation_degrees,
        )


@dataclass(frozen=True)
class RasterizedLayout:
    alpha: np.ndarray = field(repr=False, compare=False)
    shaping_succeeded: bool
    glyph_coverage_complete: bool
    clipped: bool
    shaped_runs: tuple[ShapedFontRun, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutRequest:
    text: str
    safe_region: SafeRegionArtifacts
    fonts: tuple[FontChoice, ...]
    source_font_size: float
    source_center: tuple[float, float]
    source_angle_degrees: float = 0.0
    source_weight: int = 400
    hard_font_floor: int = 10
    max_lines: int = 4
    directions: tuple[LayoutDirection, ...] = (
        LayoutDirection.VERTICAL,
        LayoutDirection.HORIZONTAL,
    )
    line_gap_options: tuple[float, ...] = (1.0, 1.12)
    tracking_options: tuple[float, ...] = (0.0, 0.1, 0.2, 0.25)
    rotation_options: tuple[float, ...] | None = None
    neighbor_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    minimum_containment: float = 0.995
    beam_width: int = 4096
    preferred_grapheme_breaks: tuple[int, ...] = ()


@dataclass(frozen=True)
class AcceptedLayout:
    candidate: LayoutCandidate
    alpha: np.ndarray = field(repr=False, compare=False)
    containment: float
    score: float
    plan_hash: str
    shaped_runs: tuple[ShapedFontRun, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutOverflow:
    available_size: tuple[int, int]
    grapheme_count: int
    suggested_max_graphemes: int
    suggested_max_lines: int
    reason: str
    rejected: tuple[tuple[str, int], ...]


LayoutResult = AcceptedLayout | LayoutOverflow
