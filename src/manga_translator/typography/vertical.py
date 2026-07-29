"""UAX #50-oriented policy for vertical Traditional Chinese layout."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum

import regex

from ..text import grapheme_clusters
from .breaking import CLOSING_PUNCTUATION, OPENING_PUNCTUATION


class VerticalOrientation(StrEnum):
    UPRIGHT = "U"
    ROTATED = "R"
    TRANSFORMED_UPRIGHT = "Tu"
    TATE_CHU_YOKO = "tcy"


@dataclass(frozen=True)
class VerticalRun:
    text: str
    orientation: VerticalOrientation
    grapheme_start: int
    grapheme_end: int


def vertical_orientation(grapheme: str) -> VerticalOrientation:
    """Return the UAX #50 class needed by the renderer for one grapheme."""

    base = grapheme[0]
    if base in OPENING_PUNCTUATION or base in CLOSING_PUNCTUATION:
        return VerticalOrientation.TRANSFORMED_UPRIGHT
    if regex.search(r"\p{Extended_Pictographic}", grapheme):
        return VerticalOrientation.UPRIGHT
    codepoint = ord(base)
    if (
        0x2E80 <= codepoint <= 0xA4CF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x3FFFF
    ):
        return VerticalOrientation.UPRIGHT
    if unicodedata.east_asian_width(base) in {"W", "F"}:
        return VerticalOrientation.UPRIGHT
    return VerticalOrientation.ROTATED


def vertical_runs(text: str) -> tuple[VerticalRun, ...]:
    """Group vertical-orientation runs and identify 2–4 character tcy candidates."""

    clusters = grapheme_clusters(text)
    runs: list[VerticalRun] = []
    cursor = 0
    while cursor < len(clusters):
        end = cursor
        while end < len(clusters) and len(clusters[end]) == 1 and clusters[end].isascii():
            end += 1
        ascii_text = "".join(clusters[cursor:end])
        candidate_length = 0
        if 2 <= len(ascii_text) <= 4 and (ascii_text.isdigit() or ascii_text.isalpha()):
            candidate_length = end - cursor
        if candidate_length:
            runs.append(
                VerticalRun(
                    ascii_text,
                    VerticalOrientation.TATE_CHU_YOKO,
                    cursor,
                    end,
                )
            )
            cursor = end
            continue

        orientation = vertical_orientation(clusters[cursor])
        end = cursor + 1
        while end < len(clusters):
            if vertical_orientation(clusters[end]) is not orientation:
                break
            end += 1
        runs.append(VerticalRun("".join(clusters[cursor:end]), orientation, cursor, end))
        cursor = end
    return tuple(runs)
