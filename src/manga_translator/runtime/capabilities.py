"""Explicit runtime capability detection for production text shaping."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import features


@dataclass(frozen=True)
class PillowCapabilities:
    freetype_version: str | None
    raqm: bool
    raqm_version: str | None
    harfbuzz_version: str | None
    fribidi_version: str | None

    @property
    def production_shaping(self) -> bool:
        return bool(
            self.freetype_version
            and self.raqm
            and self.harfbuzz_version
            and self.fribidi_version
        )


class ShapingCapabilityError(RuntimeError):
    code = "raqm_shaping_unavailable"


def pillow_capabilities() -> PillowCapabilities:
    return PillowCapabilities(
        freetype_version=features.version("freetype2"),
        raqm=bool(features.check("raqm")),
        raqm_version=features.version("raqm"),
        harfbuzz_version=features.version("harfbuzz"),
        fribidi_version=features.version("fribidi"),
    )


def require_raqm(capabilities: PillowCapabilities | None = None) -> PillowCapabilities:
    resolved = capabilities or pillow_capabilities()
    if not resolved.production_shaping:
        raise ShapingCapabilityError(
            "new typography requires Pillow FreeType+RAQM+HarfBuzz+FriBiDi; "
            f"resolved={resolved}"
        )
    return resolved
