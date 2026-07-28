"""幾何工具：bbox 相交、包含、合併、距離。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol

from shapely.geometry import Polygon as ShapelyPolygon


class HasBbox(Protocol):
    x: int
    y: int
    w: int
    h: int


def intersection_area(a: HasBbox, b: HasBbox) -> int:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def area(a: HasBbox) -> int:
    return max(0, a.w) * max(0, a.h)


def iou(a: HasBbox, b: HasBbox) -> float:
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def iom(a: HasBbox, b: HasBbox) -> float:
    inter = intersection_area(a, b)
    denom = max(1, min(area(a), area(b)))
    return inter / denom


def containment_ratio(inner: HasBbox, outer: HasBbox) -> float:
    inter = intersection_area(inner, outer)
    denom = max(1, area(inner))
    return inter / denom


def merge_bbox(items: Iterable[HasBbox]) -> tuple[int, int, int, int]:
    items = list(items)
    if not items:
        return (0, 0, 0, 0)

    x1 = min(r.x for r in items)
    y1 = min(r.y for r in items)
    x2 = max(r.x + r.w for r in items)
    y2 = max(r.y + r.h for r in items)
    return (x1, y1, x2 - x1, y2 - y1)


def center_distance(a: HasBbox, b: HasBbox) -> float:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def bbox_touch_or_near(a: HasBbox, b: HasBbox, pad: int = 0) -> bool:
    ax1, ay1 = a.x - pad, a.y - pad
    ax2, ay2 = a.x + a.w + pad, a.y + a.h + pad
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def canonical_page_polygon(
    points: Iterable[Iterable[float]], *, page_width: int, page_height: int
) -> tuple[tuple[float, float], ...]:
    """Validate and normalize a detector polygon without quantizing page coordinates."""

    canonical = tuple((float(point[0]), float(point[1])) for point in points)
    if len(canonical) < 3 or len(set(canonical)) < 3:
        raise ValueError("degenerate_polygon")
    if any(not math.isfinite(value) for point in canonical for value in point):
        raise ValueError("non_finite_polygon")
    if any(x < 0 or y < 0 or x > page_width or y > page_height for x, y in canonical):
        raise ValueError("out_of_bounds_polygon")
    shape = ShapelyPolygon(canonical)
    if shape.convex_hull.area <= 1e-9:
        raise ValueError("degenerate_polygon")
    if not shape.is_valid:
        raise ValueError("self_intersecting_polygon")
    if shape.area <= 1e-9:
        raise ValueError("degenerate_polygon")
    return canonical


def clipped_raster_bbox(
    xyxy: Iterable[float], *, page_width: int, page_height: int
) -> tuple[int, int, int, int]:
    """Preserve legacy integer raster semantics while clipping to the page."""

    x1, y1, x2, y2 = [int(value) for value in xyxy]
    x1 = max(0, min(x1, page_width - 1))
    y1 = max(0, min(y1, page_height - 1))
    x2 = max(0, min(x2, page_width))
    y2 = max(0, min(y2, page_height))
    return x1, y1, x2 - x1, y2 - y1
