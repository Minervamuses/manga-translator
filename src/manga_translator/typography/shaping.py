"""RAQM-backed full-run shaping with grapheme-safe font fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image, ImageDraw, ImageFont

from ..runtime.capabilities import require_raqm
from .fonts import FontResolver, FontRole, FontRun

Direction = Literal["ltr", "ttb"]


@dataclass(frozen=True)
class ShapedFontRun:
    text: str
    font_sha256: str
    font_path: str
    glyph_coverage: tuple[int, ...]
    direction: Direction
    language: str
    features: tuple[str, ...]
    bbox: tuple[float, float, float, float]
    advance: float
    anchor: tuple[float, float]


class LayoutEngine(Protocol):
    def measure(
        self,
        run: FontRun,
        *,
        size: int,
        direction: Direction,
        language: str,
        features: tuple[str, ...],
        stroke_width: int,
    ) -> tuple[tuple[float, float, float, float], float]: ...

    def render(
        self,
        image: Image.Image,
        run: ShapedFontRun,
        *,
        size: int,
        fill: tuple[int, int, int, int],
        stroke_width: int,
    ) -> None: ...


class PillowRaqmEngine:
    def __init__(self) -> None:
        require_raqm()

    @staticmethod
    def _font(run: FontRun, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(run.font.path), size, layout_engine=ImageFont.Layout.RAQM)

    def measure(
        self,
        run: FontRun,
        *,
        size: int,
        direction: Direction,
        language: str,
        features: tuple[str, ...],
        stroke_width: int,
    ) -> tuple[tuple[float, float, float, float], float]:
        font = self._font(run, size)
        draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        kwargs = {
            "font": font,
            "direction": direction,
            "language": language,
            "features": list(features),
            "stroke_width": stroke_width,
            "anchor": "lt",
        }
        bbox = tuple(float(value) for value in draw.textbbox((0, 0), run.text, **kwargs))
        advance = float(
            draw.textlength(
                run.text,
                font=font,
                direction=direction,
                language=language,
                features=list(features),
            )
        )
        return bbox, advance

    def render(
        self,
        image: Image.Image,
        run: ShapedFontRun,
        *,
        size: int,
        fill: tuple[int, int, int, int],
        stroke_width: int,
    ) -> None:
        font = ImageFont.truetype(run.font_path, size, layout_engine=ImageFont.Layout.RAQM)
        ImageDraw.Draw(image).text(
            run.anchor,
            run.text,
            font=font,
            fill=fill,
            direction=run.direction,
            language=run.language,
            features=list(run.features),
            stroke_width=stroke_width,
            anchor="lt",
        )


class RunShaper:
    def __init__(self, resolver: FontResolver, engine: LayoutEngine | None = None) -> None:
        self.resolver = resolver
        self.engine = engine or PillowRaqmEngine()

    def shape(
        self,
        text: str,
        *,
        role: FontRole,
        size: int,
        direction: Direction,
        stroke_width: int = 0,
    ) -> tuple[ShapedFontRun, ...]:
        language = "zh-Hant-TW"
        features: tuple[str, ...] = ()
        anchor_x = anchor_y = 0.0
        shaped: list[ShapedFontRun] = []
        for run in self.resolver.fallback_runs(text, role):
            bbox, advance = self.engine.measure(
                run,
                size=size,
                direction=direction,
                language=language,
                features=features,
                stroke_width=stroke_width,
            )
            shaped.append(
                ShapedFontRun(
                    text=run.text,
                    font_sha256=run.font.sha256,
                    font_path=str(run.font.path),
                    glyph_coverage=tuple(sorted({ord(char) for char in run.text})),
                    direction=direction,
                    language=language,
                    features=features,
                    bbox=bbox,
                    advance=advance,
                    anchor=(anchor_x, anchor_y),
                )
            )
            if direction == "ttb":
                anchor_y += advance
            else:
                anchor_x += advance
        return tuple(shaped)
