"""閱讀順序排序工具。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar


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
