"""Typed style estimates with explicit uncertainty."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class StyleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Estimate(StyleModel, Generic[T]):
    value: T | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    status: Literal["known", "unknown"]


RGB = tuple[int, int, int]


class ShadowEstimate(StyleModel):
    color: Estimate[RGB]
    offset: Estimate[tuple[float, float]]
    confidence: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    status: Literal["known", "unknown"]


class ExtractedStyle(StyleModel):
    source: Literal["original_image"] = "original_image"
    fill: Estimate[RGB]
    stroke: Estimate[RGB]
    stroke_width: Estimate[float]
    ink_density: Estimate[float]
    normalized_stroke_width: Estimate[float]
    width_height_ratio: Estimate[float]
    edge_roundness: Estimate[float]
    stroke_variation: Estimate[float]
    source_angle: Estimate[float]
    shadow: ShadowEstimate

    def renderer_values(
        self, *, min_confidence: float, default_fill: RGB, default_stroke: RGB | None
    ) -> dict[str, object]:
        """Expose only estimates that clear the renderer confidence policy."""

        return {
            "fill_rgb": (
                self.fill.value
                if self.fill.status == "known" and self.fill.confidence >= min_confidence
                else default_fill
            ),
            "stroke_rgb": (
                self.stroke.value
                if self.stroke.status == "known" and self.stroke.confidence >= min_confidence
                else default_stroke
            ),
            "stroke_width": (
                self.stroke_width.value
                if self.stroke_width.status == "known"
                and self.stroke_width.confidence >= min_confidence
                else 0.0
            ),
            "shadow": (
                self.shadow
                if self.shadow.status == "known" and self.shadow.confidence >= min_confidence
                else None
            ),
        }
