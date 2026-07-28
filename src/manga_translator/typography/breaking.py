"""Unicode line-break opportunities with Traditional Chinese tailoring."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from uniseg.linebreak import line_break_boundaries

from ..text import grapheme_clusters

OPENING_PUNCTUATION = frozenset("（［｛〔〈《「『【〖〘〚〝‘“([{")
CLOSING_PUNCTUATION = frozenset("、。，．？！：；）》」』】〕〉》〗〙〛〞〟’”）］｝,.:;!?)]}")
ELLIPSIS_AND_DASH = frozenset("…⋯—―─")
_ATOMIC_TOKEN = re.compile(
    r"(?<![0-9])[0-9]{2,4}(?![0-9])"
    r"|(?<![A-Za-z])[A-Za-z]{2,4}(?![A-Za-z])"
    r"|&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);"
)


class BreakStrength(StrEnum):
    ALLOWED = "allowed"
    MANDATORY = "mandatory"


@dataclass(frozen=True)
class BreakOpportunity:
    """A legal boundary, expressed in both code-point and grapheme indices."""

    index: int
    grapheme_index: int
    strength: BreakStrength
    preference: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BreakViolation:
    """A candidate or requested boundary rejected by a hard CLREQ rule."""

    index: int
    rule: str
    before: str
    after: str


@dataclass(frozen=True)
class LineBreakAnalysis:
    text: str
    opportunities: tuple[BreakOpportunity, ...]
    violations: tuple[BreakViolation, ...]

    @property
    def legal_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.opportunities)


def _cluster_boundaries(text: str) -> tuple[tuple[str, ...], dict[int, int]]:
    clusters = grapheme_clusters(text)
    offsets: dict[int, int] = {0: 0}
    cursor = 0
    for grapheme_index, cluster in enumerate(clusters, start=1):
        cursor += len(cluster)
        offsets[cursor] = grapheme_index
    return clusters, offsets


def _automatic_atomic_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(match.span() for match in _ATOMIC_TOKEN.finditer(text))


def _hard_rule(
    text: str,
    index: int,
    clusters: tuple[str, ...],
    grapheme_index: int,
    atomic_spans: tuple[tuple[int, int], ...],
) -> str | None:
    if index <= 0 or index >= len(text):
        return None
    before = clusters[grapheme_index - 1]
    after = clusters[grapheme_index]
    if after[0] in CLOSING_PUNCTUATION:
        return "clreq_no_closing_punctuation_at_line_start"
    if before[-1] in OPENING_PUNCTUATION:
        return "clreq_no_opening_punctuation_at_line_end"
    if before[-1] in ELLIPSIS_AND_DASH and after[0] in ELLIPSIS_AND_DASH:
        return "clreq_keep_ellipsis_or_dash_sequence"
    if any(start < index < end for start, end in atomic_spans):
        return "clreq_keep_short_token_or_entity"
    return None


def _preference(before: str, after: str) -> tuple[int, tuple[str, ...]]:
    if before.isspace():
        return 60, ("space",)
    if before[-1] in CLOSING_PUNCTUATION:
        return 40, ("after_punctuation",)
    if after.isspace():
        return 20, ("before_space",)
    return 0, ("uax14",)


def analyze_line_breaks(
    text: str,
    *,
    atomic_spans: Iterable[tuple[int, int]] = (),
    preferred_grapheme_breaks: Iterable[int] = (),
) -> LineBreakAnalysis:
    """Return UAX #14 opportunities after applying hard CLREQ tailoring.

    The input is never normalized or stripped. Explicit CR, LF, and CRLF
    graphemes become mandatory opportunities, while the remaining legal
    boundaries retain a small preference score for the later layout solver.
    """

    clusters, offsets = _cluster_boundaries(text)
    spans = (*_automatic_atomic_spans(text), *tuple(atomic_spans))
    preferred_breaks = frozenset(preferred_grapheme_breaks)
    base_boundaries = set(line_break_boundaries(text))
    opportunities: list[BreakOpportunity] = []
    violations: list[BreakViolation] = []

    for index in sorted(base_boundaries):
        grapheme_index = offsets.get(index)
        if grapheme_index is None:
            continue
        previous = clusters[grapheme_index - 1] if grapheme_index else ""
        following = clusters[grapheme_index] if grapheme_index < len(clusters) else ""
        mandatory = previous in {"\n", "\r", "\r\n"}
        rule = _hard_rule(text, index, clusters, grapheme_index, spans)
        if rule is not None and not mandatory:
            violations.append(BreakViolation(index, rule, previous, following))
            continue
        preference, reasons = _preference(previous, following)
        if mandatory:
            preference, reasons = 1_000, ("explicit_newline",)
        elif grapheme_index in preferred_breaks:
            preference += 100
            reasons = (*reasons, "preferred_break")
        opportunities.append(
            BreakOpportunity(
                index=index,
                grapheme_index=grapheme_index,
                strength=BreakStrength.MANDATORY if mandatory else BreakStrength.ALLOWED,
                preference=preference,
                reasons=reasons,
            )
        )

    reported = {(item.index, item.rule) for item in violations}
    for index, grapheme_index in sorted(offsets.items()):
        rule = _hard_rule(text, index, clusters, grapheme_index, spans)
        if rule is None or (index, rule) in reported:
            continue
        before = clusters[grapheme_index - 1] if grapheme_index else ""
        after = clusters[grapheme_index] if grapheme_index < len(clusters) else ""
        violations.append(BreakViolation(index, rule, before, after))
    return LineBreakAnalysis(text, tuple(opportunities), tuple(violations))


def validate_breaks(
    text: str,
    indices: Iterable[int],
    *,
    atomic_spans: Iterable[tuple[int, int]] = (),
) -> tuple[BreakViolation, ...]:
    """Report hard-rule violations for boundaries selected by a solver."""

    clusters, offsets = _cluster_boundaries(text)
    spans = (*_automatic_atomic_spans(text), *tuple(atomic_spans))
    violations: list[BreakViolation] = []
    for index in indices:
        grapheme_index = offsets.get(index)
        if grapheme_index is None:
            violations.append(BreakViolation(index, "not_grapheme_boundary", "", ""))
            continue
        rule = _hard_rule(text, index, clusters, grapheme_index, spans)
        if rule is not None:
            before = clusters[grapheme_index - 1] if grapheme_index else ""
            after = clusters[grapheme_index] if grapheme_index < len(clusters) else ""
            violations.append(BreakViolation(index, rule, before, after))
    return tuple(violations)


def _render_slice(text: str, start: int, end: int) -> str:
    return text[start:end].removesuffix("\r\n").removesuffix("\n").removesuffix("\r")


def greedy_legal_wrap(
    text: str,
    measure: Callable[[str], float],
    max_extent: float,
    *,
    preferred_grapheme_breaks: Iterable[int] = (),
) -> tuple[str, ...]:
    """Choose only legal boundaries, preferring the furthest fitting one."""

    analysis = analyze_line_breaks(
        text,
        preferred_grapheme_breaks=preferred_grapheme_breaks,
    )
    if not text:
        return ()
    boundaries = analysis.opportunities
    lines: list[str] = []
    start = 0
    cursor = 0
    while start < len(text):
        candidates = [item for item in boundaries if item.index > start]
        if not candidates:
            lines.append(_render_slice(text, start, len(text)))
            break
        chosen: BreakOpportunity | None = None
        for item in candidates:
            candidate = _render_slice(text, start, item.index)
            if measure(candidate) <= max_extent or chosen is None:
                chosen = item
            if item.strength is BreakStrength.MANDATORY or measure(candidate) > max_extent:
                break
        assert chosen is not None
        if chosen.index <= cursor:
            raise RuntimeError("line-breaking cursor did not advance")
        lines.append(_render_slice(text, start, chosen.index))
        cursor = chosen.index
        start = chosen.index
    return tuple(lines)


def balanced_legal_chunks(
    text: str,
    count: int,
    *,
    preferred_grapheme_breaks: Iterable[int] = (),
) -> tuple[str, ...]:
    """Split near equal grapheme counts without inventing illegal boundaries."""

    if not text:
        return ()
    analysis = analyze_line_breaks(
        text,
        preferred_grapheme_breaks=preferred_grapheme_breaks,
    )
    internal = [item for item in analysis.opportunities if item.index < len(text)]
    mandatory = {item.index for item in internal if item.strength is BreakStrength.MANDATORY}
    desired_count = max(1, count, len(mandatory) + 1)
    desired_count = min(desired_count, len(internal) + 1)
    selected = set(mandatory)
    grapheme_count = len(grapheme_clusters(text))
    for part in range(1, desired_count):
        target = grapheme_count * part / desired_count
        remaining = [item for item in internal if item.index not in selected]
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda item: (abs(item.grapheme_index - target), -item.preference),
        )
        selected.add(best.index)
    cuts = [0, *sorted(selected), len(text)]
    return tuple(_render_slice(text, start, end) for start, end in pairwise(cuts))
