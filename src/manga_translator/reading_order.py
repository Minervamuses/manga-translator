"""Deterministic, panel-aware Japanese manga reading order."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar
from uuid import UUID

from .domain.models import BoundingBox, ReadingOrderOverride
from .order.panels import PanelCandidate


class HasRect(Protocol):
    x: int
    y: int
    w: int
    h: int


T = TypeVar("T", bound=HasRect)


def _jp_vertical_key(item: HasRect) -> tuple[float, float]:
    cx = item.x + item.w / 2.0
    cy = item.y + item.h / 2.0
    return (-cx, cy)


def _ltr_horizontal_key(item: HasRect) -> tuple[float, float]:
    cx = item.x + item.w / 2.0
    cy = item.y + item.h / 2.0
    return (cy, cx)


def sort_regions_jp_vertical(regions: Iterable[T]) -> list[T]:
    return sorted(regions, key=_jp_vertical_key)


def sort_groups_jp_vertical(groups: Iterable[T]) -> list[T]:
    return sorted(groups, key=_jp_vertical_key)


def sort_groups_auto(groups: Iterable[T], vertical_ratio: float | None = None) -> list[T]:
    groups = list(groups)
    if not groups:
        return groups

    if vertical_ratio is None:
        vertical_like = sum(1 for g in groups if g.h >= g.w)
        vertical_ratio = vertical_like / max(1, len(groups))

    if vertical_ratio >= 0.5:
        return sorted(groups, key=_jp_vertical_key)
    return sorted(groups, key=_ltr_horizontal_key)


@dataclass(frozen=True, slots=True)
class OrderRegion:
    region_id: UUID
    bbox: BoundingBox
    orientation: Literal["horizontal", "vertical", "rotated", "unknown"] = "unknown"


@dataclass(frozen=True, slots=True)
class OrderedRegion:
    region_id: UUID
    panel_id: str | None
    order: int


@dataclass(frozen=True, slots=True)
class ReadingOrderIssue:
    code: Literal["order_uncertain", "precedence_cycle"]
    message: str
    region_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadingOrderResult:
    regions: tuple[OrderedRegion, ...]
    confidence: float
    order_uncertain: bool
    used_manual_override: bool
    issues: tuple[ReadingOrderIssue, ...] = ()


def _page_key(region: OrderRegion) -> tuple[float, float, str]:
    """Conservative deterministic Japanese page order used for all fallbacks."""

    center_x = region.bbox.x + region.bbox.width / 2
    center_y = region.bbox.y + region.bbox.height / 2
    return (-center_x, center_y, str(region.region_id))


def _intersection_area(a: BoundingBox, panel: PanelCandidate) -> float:
    width = max(0.0, min(a.right, panel.right) - max(a.x, panel.x))
    height = max(0.0, min(a.bottom, panel.bottom) - max(a.y, panel.y))
    return width * height


def _panel_sequence(panels: Sequence[PanelCandidate]) -> list[PanelCandidate]:
    """Top bands first; panels in an overlapping band go right-to-left."""

    remaining = sorted(panels, key=lambda item: (item.y, -item.x, item.panel_id))
    result: list[PanelCandidate] = []
    while remaining:
        anchor = remaining[0]
        band_bottom = anchor.bottom
        band: list[PanelCandidate] = []
        rest: list[PanelCandidate] = []
        for panel in remaining:
            overlap = min(band_bottom, panel.bottom) - max(anchor.y, panel.y)
            if overlap > 0 or abs(panel.y - anchor.y) <= min(anchor.height, panel.height) * 0.2:
                band.append(panel)
                band_bottom = max(band_bottom, panel.bottom)
            else:
                rest.append(panel)
        result.extend(sorted(band, key=lambda item: (-item.x, item.y, item.panel_id)))
        remaining = rest
    return result


def _stable_topological_order(
    regions: Sequence[OrderRegion], edges: Iterable[tuple[UUID, UUID]]
) -> tuple[list[OrderRegion], bool]:
    by_id = {region.region_id: region for region in regions}
    successors: dict[UUID, set[UUID]] = defaultdict(set)
    indegree = {region.region_id: 0 for region in regions}
    for before, after in edges:
        if before == after or before not in by_id or after not in by_id:
            continue
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1
    ready = sorted((by_id[key] for key, degree in indegree.items() if degree == 0), key=_page_key)
    ordered: list[OrderRegion] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for following in sorted(successors[current.region_id], key=str):
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(by_id[following])
                ready.sort(key=_page_key)
    return ordered, len(ordered) != len(regions)


def order_precedence_graph(
    regions: Sequence[OrderRegion], edges: Iterable[tuple[UUID, UUID]]
) -> tuple[tuple[OrderRegion, ...], bool]:
    """Resolve an explicit precedence graph and expose cycle detection for audit/tests."""

    ordered, cyclic = _stable_topological_order(regions, edges)
    if cyclic:
        return tuple(sorted(regions, key=_page_key)), True
    return tuple(ordered), False


def _within_panel(regions: Sequence[OrderRegion]) -> tuple[list[OrderRegion], bool]:
    if all(region.orientation == "vertical" for region in regions):
        return sorted(regions, key=_page_key), False
    if all(region.orientation == "horizontal" for region in regions):
        return (
            sorted(
                regions,
                key=lambda item: (
                    item.bbox.y + item.bbox.height / 2,
                    item.bbox.x + item.bbox.width / 2,
                    str(item.region_id),
                ),
            ),
            False,
        )

    edges: set[tuple[UUID, UUID]] = set()
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if left.bbox.bottom <= right.bbox.y:
                edges.add((left.region_id, right.region_id))
            elif right.bbox.bottom <= left.bbox.y or left.bbox.right <= right.bbox.x:
                edges.add((right.region_id, left.region_id))
            elif (
                right.bbox.right <= left.bbox.x
                or left.orientation == "vertical"
                and right.orientation != "vertical"
            ):
                edges.add((left.region_id, right.region_id))
            elif right.orientation == "vertical" and left.orientation != "vertical":
                edges.add((right.region_id, left.region_id))
    ordered, cyclic = _stable_topological_order(regions, edges)
    return ordered, cyclic


def _manual_result(
    regions: Sequence[OrderRegion], overrides: Sequence[ReadingOrderOverride]
) -> ReadingOrderResult | None:
    if not overrides:
        return None
    by_region = {override.region_id: override for override in overrides}
    if set(by_region) != {region.region_id for region in regions}:
        return None
    ordered = sorted(regions, key=lambda region: by_region[region.region_id].order)
    return ReadingOrderResult(
        regions=tuple(
            OrderedRegion(region.region_id, by_region[region.region_id].panel_id, index)
            for index, region in enumerate(ordered)
        ),
        confidence=1.0,
        order_uncertain=False,
        used_manual_override=True,
    )


def _fallback(
    regions: Sequence[OrderRegion], message: str, *, code: str = "order_uncertain"
) -> ReadingOrderResult:
    ordered = sorted(regions, key=_page_key)
    issue_code: Literal["order_uncertain", "precedence_cycle"] = (
        "precedence_cycle" if code == "precedence_cycle" else "order_uncertain"
    )
    return ReadingOrderResult(
        regions=tuple(
            OrderedRegion(region.region_id, None, index) for index, region in enumerate(ordered)
        ),
        confidence=0.0,
        order_uncertain=True,
        used_manual_override=False,
        issues=(
            ReadingOrderIssue(issue_code, message, tuple(region.region_id for region in regions)),
        ),
    )


def resolve_reading_order(
    regions: Sequence[OrderRegion],
    *,
    panels: Sequence[PanelCandidate] = (),
    manual_overrides: Sequence[ReadingOrderOverride] = (),
) -> ReadingOrderResult:
    """Resolve final order without treating ambiguous panel inference as fact."""

    manual = _manual_result(regions, manual_overrides)
    if manual is not None:
        return manual
    if manual_overrides:
        return _fallback(regions, "manual reading-order override does not cover every region")
    if not regions:
        return ReadingOrderResult((), 1.0, False, False)
    if not panels or min(panel.confidence for panel in panels) < 0.55:
        return _fallback(regions, "no sufficiently confident panel layout")

    assignments: dict[str, list[OrderRegion]] = {panel.panel_id: [] for panel in panels}
    for region in regions:
        region_area = region.bbox.width * region.bbox.height
        matches = [
            panel
            for panel in panels
            if _intersection_area(region.bbox, panel) / max(1.0, region_area) >= 0.2
        ]
        center_matches = [
            panel
            for panel in matches
            if panel.x <= region.bbox.x + region.bbox.width / 2 <= panel.right
            and panel.y <= region.bbox.y + region.bbox.height / 2 <= panel.bottom
        ]
        if len(matches) > 1 or len(center_matches) != 1:
            return _fallback(
                regions, f"region {region.region_id} crosses or misses panel boundaries"
            )
        assignments[center_matches[0].panel_id].append(region)

    ordered: list[tuple[OrderRegion, str]] = []
    for panel in _panel_sequence(panels):
        members = assignments[panel.panel_id]
        panel_order, cyclic = _within_panel(members)
        if cyclic:
            return _fallback(
                regions,
                f"mixed-orientation precedence cycle in panel {panel.panel_id}",
                code="precedence_cycle",
            )
        ordered.extend((region, panel.panel_id) for region in panel_order)
    if len(ordered) != len(regions):
        return _fallback(regions, "panel assignment did not preserve every region")
    return ReadingOrderResult(
        regions=tuple(
            OrderedRegion(region.region_id, panel_id, index)
            for index, (region, panel_id) in enumerate(ordered)
        ),
        confidence=min(panel.confidence for panel in panels),
        order_uncertain=False,
        used_manual_override=False,
    )
