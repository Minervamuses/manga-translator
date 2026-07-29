"""Lossless text variants and auditable transformations."""

from __future__ import annotations

import regex
from pydantic import BaseModel, ConfigDict


class Transformation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str
    before: str
    after: str


class TextVariants(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    raw: str
    nfc_display: str
    nfkc_comparison_key: str
    transformations: tuple[Transformation, ...] = ()
    preferred_breaks: tuple[int, ...] = ()

    def reconstruct_raw(self) -> str:
        return self.raw


def grapheme_clusters(text: str) -> tuple[str, ...]:
    """Segment extended grapheme clusters according to UAX #29."""

    return tuple(regex.findall(r"\X", text))
