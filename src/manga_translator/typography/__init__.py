"""Font catalog, shaping, breaking, and layout primitives."""

from .breaking import LineBreakAnalysis, analyze_line_breaks
from .fonts import FontCatalog, FontResolver
from .safe_region import SafeRegionArtifacts, build_safe_region
from .vertical import VerticalOrientation, vertical_runs

__all__ = [
    "FontCatalog",
    "FontResolver",
    "LineBreakAnalysis",
    "SafeRegionArtifacts",
    "VerticalOrientation",
    "analyze_line_breaks",
    "build_safe_region",
    "vertical_runs",
]
