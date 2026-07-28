"""Safe display normalization that never overwrites raw text."""

from __future__ import annotations

import unicodedata

from .variants import TextVariants, Transformation, grapheme_clusters


def _apply(value: str, rule_id: str, transform, log: list[Transformation]) -> str:
    updated = transform(value)
    if updated != value:
        log.append(Transformation(rule_id=rule_id, before=value, after=updated))
    return updated


def normalize_text(raw: str, *, preferred_break_markers: tuple[str, ...] = ()) -> TextVariants:
    transformations: list[Transformation] = []
    display = _apply(raw, "line_endings.lf", lambda value: value.replace("\r\n", "\n").replace("\r", "\n"), transformations)
    display = _apply(
        display,
        "controls.remove_explicit",
        lambda value: "".join(
            char
            for char in value
            if char in {"\n", "\t"}
            or (char != "\ufffd" and unicodedata.category(char) not in {"Cc", "Cs", "Co"})
        ),
        transformations,
    )
    display = _apply(display, "outer_whitespace.strip", str.strip, transformations)
    display = _apply(display, "unicode.nfc", lambda value: unicodedata.normalize("NFC", value), transformations)
    clusters = grapheme_clusters(display)
    preferred_breaks = tuple(
        index + 1
        for index, cluster in enumerate(clusters)
        if cluster in preferred_break_markers or cluster == "\n"
    )
    return TextVariants(
        raw=raw,
        nfc_display=display,
        nfkc_comparison_key=unicodedata.normalize("NFKC", display).casefold(),
        transformations=tuple(transformations),
        preferred_breaks=preferred_breaks,
    )
