"""Conservative panel proposals from borders and recursive white-gutter XY cuts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

import cv2
import numpy as np


class HasBox(Protocol):
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PanelCandidate:
    panel_id: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: Literal["border", "xy_cut", "manual"]

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


def _panel_id(x: float, y: float, width: float, height: float) -> str:
    value = f"{x:.3f}:{y:.3f}:{width:.3f}:{height:.3f}".encode()
    return f"panel-{sha256(value).hexdigest()[:12]}"


def _area(box: PanelCandidate | HasBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def _intersection(a: PanelCandidate | HasBox, b: PanelCandidate | HasBox) -> float:
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    return max(0.0, right - max(a.x, b.x)) * max(0.0, bottom - max(a.y, b.y))


def _iou(a: PanelCandidate, b: PanelCandidate) -> float:
    overlap = _intersection(a, b)
    return overlap / max(1.0, _area(a) + _area(b) - overlap)


def _border_candidates(gray: np.ndarray) -> list[PanelCandidate]:
    height, width = gray.shape
    page_area = float(width * height)
    # Manga borders are dark strokes around substantially white interiors. Closing
    # connects small breaks without filling the speech/text content itself.
    dark = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
    closed = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: list[PanelCandidate] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        fraction = (box_width * box_height) / page_area
        if fraction < 0.04 or fraction > 0.92 or box_width < 12 or box_height < 12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        rectangularity = min(1.0, cv2.contourArea(contour) / max(1.0, box_width * box_height))
        if len(approximation) > 8 and rectangularity < 0.55:
            continue
        confidence = min(0.98, 0.62 + 0.25 * rectangularity)
        found.append(
            PanelCandidate(
                _panel_id(x, y, box_width, box_height),
                float(x),
                float(y),
                float(box_width),
                float(box_height),
                confidence,
                "border",
            )
        )
    return found


def _longest_true_run(values: np.ndarray) -> tuple[int, int]:
    best_start = best_length = current_start = current_length = 0
    for index, value in enumerate(values.tolist() + [False]):
        if value:
            if current_length == 0:
                current_start = index
            current_length += 1
        else:
            if current_length > best_length:
                best_start, best_length = current_start, current_length
            current_length = 0
    return best_start, best_length


def _xy_cut(
    occupied: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    minimum_gutter: int,
    minimum_extent: int,
) -> list[tuple[int, int, int, int]]:
    x, y, width, height = bounds
    if width < minimum_extent * 2 or height < minimum_extent * 2:
        return [bounds]
    crop = occupied[y : y + height, x : x + width]
    empty_columns = np.mean(crop, axis=0) <= 0.002
    empty_rows = np.mean(crop, axis=1) <= 0.002
    col_start, col_length = _longest_true_run(empty_columns)
    row_start, row_length = _longest_true_run(empty_rows)

    vertical_ok = (
        col_length >= minimum_gutter
        and col_start >= minimum_extent
        and width - col_start - col_length >= minimum_extent
    )
    horizontal_ok = (
        row_length >= minimum_gutter
        and row_start >= minimum_extent
        and height - row_start - row_length >= minimum_extent
    )
    if not vertical_ok and not horizontal_ok:
        return [bounds]
    use_vertical = vertical_ok and (not horizontal_ok or col_length / width >= row_length / height)
    if use_vertical:
        left = (x, y, col_start, height)
        right_x = x + col_start + col_length
        right = (right_x, y, x + width - right_x, height)
        return _xy_cut(
            occupied, left, minimum_gutter=minimum_gutter, minimum_extent=minimum_extent
        ) + _xy_cut(occupied, right, minimum_gutter=minimum_gutter, minimum_extent=minimum_extent)
    top = (x, y, width, row_start)
    bottom_y = y + row_start + row_length
    bottom = (x, bottom_y, width, y + height - bottom_y)
    return _xy_cut(
        occupied, top, minimum_gutter=minimum_gutter, minimum_extent=minimum_extent
    ) + _xy_cut(occupied, bottom, minimum_gutter=minimum_gutter, minimum_extent=minimum_extent)


def _xy_candidates(gray: np.ndarray) -> list[PanelCandidate]:
    height, width = gray.shape
    occupied = gray < 245
    gutter = max(4, round(min(width, height) * 0.025))
    extent = max(12, round(min(width, height) * 0.12))
    boxes = _xy_cut(
        occupied,
        (0, 0, width, height),
        minimum_gutter=gutter,
        minimum_extent=extent,
    )
    if len(boxes) <= 1:
        return []
    return [
        PanelCandidate(
            _panel_id(x, y, box_width, box_height),
            float(x),
            float(y),
            float(box_width),
            float(box_height),
            0.72,
            "xy_cut",
        )
        for x, y, box_width, box_height in boxes
        if box_width * box_height >= width * height * 0.035
    ]


def _coverage(candidate: PanelCandidate, regions: Sequence[HasBox]) -> float:
    if not regions:
        return 1.0
    contained = sum(
        1
        for region in regions
        if candidate.x <= region.x + region.width / 2 <= candidate.right
        and candidate.y <= region.y + region.height / 2 <= candidate.bottom
    )
    return contained / len(regions)


def detect_panel_candidates(
    image: np.ndarray,
    regions: Sequence[HasBox] = (),
    *,
    minimum_confidence: float = 0.55,
) -> tuple[PanelCandidate, ...]:
    """Return separated panel proposals; weak/duplicate geometry is discarded."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("panel detection expects a 2D or 3D image")
    if gray.size == 0:
        return ()

    proposals = sorted(
        [*_border_candidates(gray), *_xy_candidates(gray)],
        key=lambda item: (-item.confidence, -_area(item), item.y, item.x),
    )
    accepted: list[PanelCandidate] = []
    for proposal in proposals:
        if proposal.confidence < minimum_confidence:
            continue
        if regions and _coverage(proposal, regions) == 0:
            continue
        if any(_iou(proposal, current) > 0.72 for current in accepted):
            continue
        # Nested contours are usually speech balloons or an inner/outer border pair.
        if any(
            _intersection(proposal, current) / max(1.0, min(_area(proposal), _area(current))) > 0.88
            for current in accepted
        ):
            continue
        accepted.append(proposal)

    if regions:
        covered = {
            index
            for index, region in enumerate(regions)
            if any(
                panel.x <= region.x + region.width / 2 <= panel.right
                and panel.y <= region.y + region.height / 2 <= panel.bottom
                for panel in accepted
            )
        }
        if len(covered) / len(regions) < 0.6:
            return ()
    return tuple(sorted(accepted, key=lambda item: (item.y, -item.x, item.panel_id)))
