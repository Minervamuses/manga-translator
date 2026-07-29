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
    background: Estimate[RGB] | None = None
    ink_density: Estimate[float]
    normalized_stroke_width: Estimate[float]
    width_height_ratio: Estimate[float]
    edge_roundness: Estimate[float]
    stroke_variation: Estimate[float]
    source_angle: Estimate[float]
    shadow: ShadowEstimate

    def renderer_values(
        self,
        *,
        min_confidence: float,
        default_fill: RGB,
        default_stroke: RGB | None,
        stroke_min_confidence: float | None = None,
        minimum_stroke_contrast: float = 0.0,
    ) -> dict[str, object]:
        """Expose only estimates that clear the renderer confidence policy."""

        fill = (
            self.fill.value
            if self.fill.status == "known" and self.fill.confidence >= min_confidence
            else default_fill
        )
        stroke_threshold = (
            min_confidence
            if stroke_min_confidence is None
            else stroke_min_confidence
        )
        stroke_contrast = (
            max(abs(int(channel) - int(fill[index])) for index, channel in enumerate(self.stroke.value))
            if self.stroke.value is not None
            else 0.0
        )
        use_stroke = (
            self.stroke.status == "known"
            and self.stroke.value is not None
            and self.stroke.confidence >= stroke_threshold
            and self.stroke_width.status == "known"
            and self.stroke_width.value is not None
            and self.stroke_width.confidence >= stroke_threshold
            and stroke_contrast >= minimum_stroke_contrast
        )
        return {
            "fill_rgb": fill,
            "stroke_rgb": self.stroke.value if use_stroke else default_stroke,
            "stroke_width": self.stroke_width.value if use_stroke else 0.0,
            "shadow": (
                self.shadow
                if self.shadow.status == "known" and self.shadow.confidence >= min_confidence
                else None
            ),
        }
