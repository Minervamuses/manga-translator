"""Font catalog, shaping, breaking, and layout primitives."""

from .breaking import LineBreakAnalysis, analyze_line_breaks
from .fonts import FontCatalog, FontResolver
from .layout import AcceptedLayout, LayoutOverflow, LayoutRequest
from .safe_region import SafeRegionArtifacts, build_safe_region
from .solver import PillowLayoutRasterizer, solve_layout
from .vertical import VerticalOrientation, vertical_runs

__all__ = [
    "AcceptedLayout",
    "FontCatalog",
    "FontResolver",
    "LayoutOverflow",
    "LayoutRequest",
    "LineBreakAnalysis",
    "PillowLayoutRasterizer",
    "SafeRegionArtifacts",
    "VerticalOrientation",
    "analyze_line_breaks",
    "build_safe_region",
    "solve_layout",
    "vertical_runs",
]
