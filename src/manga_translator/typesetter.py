"""漫畫排版：重建原文字塊幾何、維持可讀字級，再安全寫回局部 ROI。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import TypesettingConfig
from .detector import TextGroup, TextRegion
from .text import grapheme_clusters, normalize_text
from .typography.breaking import balanced_legal_chunks, greedy_legal_wrap
from .typography.safe_region import build_safe_region


@dataclass(frozen=True)
class OriginalTextGeometry:
    """Original text-block geometry inferred from detector masks and regions."""

    bbox: tuple[int, int, int, int]
    font_size: float
    primary_count: int
    primary_step: float
    secondary_step: float
    source_length: int


@dataclass(frozen=True)
class TextLayoutPlan:
    """A deterministic, preflighted layout used both before and after inpainting."""

    bbox: tuple[int, int, int, int]
    direction: str
    font_size: int
    chunks: tuple[str, ...]
    primary_step: float
    secondary_step: float
    center_x: float
    center_y: float
    block_width: float
    block_height: float
    fits: bool = True
    reason: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        x, y, w, h = self.bbox
        return {
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "direction": self.direction,
            "font_size": self.font_size,
            "chunks": list(self.chunks),
            "primary_step": round(float(self.primary_step), 3),
            "secondary_step": round(float(self.secondary_step), 3),
            "center_x": round(float(self.center_x), 3),
            "center_y": round(float(self.center_y), 3),
            "block_width": round(float(self.block_width), 3),
            "block_height": round(float(self.block_height), 3),
            "fits": self.fits,
            "reason": self.reason,
            "score": round(float(self.score), 5),
        }


@lru_cache(maxsize=256)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到字體檔：{path}")
    return ImageFont.truetype(str(path), size)


def _get_font_metrics(font: ImageFont.FreeTypeFont) -> tuple[int, int, int]:
    ascent, descent = font.getmetrics()
    line_height = max(1, int(ascent + abs(descent)))
    return int(ascent), int(descent), line_height


def _measure_char_advance(font: ImageFont.FreeTypeFont, char: str) -> int:
    try:
        return max(1, round(font.getlength(char)))
    except (OSError, TypeError, ValueError):
        bbox = font.getbbox(char)
        return max(1, int(bbox[2] - bbox[0]))


def _has_glyph(font_path: str, size: int, char: str) -> bool:
    del size
    from .typography.fonts import font_has_glyph

    return font_has_glyph(font_path, char)


def _get_font_and_char(
    char: str,
    size: int,
    primary_font_path: str,
    fallback_font_path: str | None = None,
    replace_unsupported: bool = True,
) -> tuple[ImageFont.FreeTypeFont, str]:
    if _has_glyph(primary_font_path, size, char):
        return _load_font(primary_font_path, size), char
    if fallback_font_path and _has_glyph(fallback_font_path, size, char):
        return _load_font(fallback_font_path, size), char

    if replace_unsupported:
        for replacement in ("□", "?", "・"):
            if fallback_font_path and _has_glyph(fallback_font_path, size, replacement):
                return _load_font(fallback_font_path, size), replacement
            if _has_glyph(primary_font_path, size, replacement):
                return _load_font(primary_font_path, size), replacement
    return _load_font(primary_font_path, size), char


def _sanitize_render_text(text: str) -> str:
    return normalize_text(text or "").nfc_display


def _visible_length(text: str) -> int:
    return sum(not cluster.isspace() for cluster in grapheme_clusters(_sanitize_render_text(text)))


def _decide_direction(obj: TextRegion | TextGroup, config_dir: str) -> str:
    if config_dir != "auto":
        return config_dir
    if hasattr(obj, "vertical"):
        return "vertical" if bool(getattr(obj, "vertical", False)) else "horizontal"
    return "vertical" if obj.h > obj.w * 1.2 else "horizontal"


def _wrap_horizontal(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    def measure(value: str) -> float:
        bbox = font.getbbox(value)
        return float(bbox[2] - bbox[0])

    return list(greedy_legal_wrap(text, measure, max_w))


def _font_bounds(cfg: TypesettingConfig, preferred_font_size: float | None) -> tuple[int, int]:
    if preferred_font_size is None or preferred_font_size <= 0:
        return cfg.font_size_min, cfg.font_size_max
    preferred = preferred_font_size * cfg.font_size_scale
    low_scale = cfg.min_font_scale if cfg.reject_unreadable_layout else cfg.hard_min_font_scale
    high = min(
        cfg.font_size_max,
        max(cfg.font_size_min, math.ceil(preferred * cfg.max_font_growth_ratio)),
    )
    low = min(high, max(cfg.font_size_min, math.floor(preferred * low_scale)))
    return low, high


def _calculate_font_size(
    text: str,
    available_w: int,
    available_h: int,
    font_path: str,
    direction: str,
    cfg: TypesettingConfig,
    preferred_font_size: float | None = None,
) -> int:
    """Compatibility helper: largest readable size that actually fits.

    The old implementation silently searched down to 4 px.  The lower bound is
    now tied to the original font estimate; callers that need a hard fit should
    use :func:`plan_text_layout` and handle an unrenderable plan explicitly.
    """

    if available_w <= 0 or available_h <= 0 or not text:
        return cfg.font_size_min

    low, high = _font_bounds(cfg, preferred_font_size)
    best: int | None = None
    while low <= high:
        mid = (low + high) // 2
        font = _load_font(font_path, mid)
        _ascent, _descent, line_h = _get_font_metrics(font)
        if direction == "vertical":
            char_step = max(1, round(line_h * cfg.vertical_char_spacing))
            col_step = max(1, round(mid * cfg.line_spacing))
            chars_per_col = max(1, available_h // char_step)
            cols_needed = math.ceil(len(grapheme_clusters(text)) / chars_per_col)
            fits = mid + max(0, cols_needed - 1) * col_step <= available_w
        else:
            lines = _wrap_horizontal(text, font, available_w)
            line_step = max(1, round(line_h * cfg.line_spacing))
            total_h = line_h + max(0, len(lines) - 1) * line_step
            fits = total_h <= available_h
        if fits:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best if best is not None else _font_bounds(cfg, preferred_font_size)[0]


def _build_group_local_mask(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    bx, by, bw, bh = bbox
    local = np.zeros((bh, bw), dtype=np.uint8)

    def paste(source: np.ndarray, source_bbox: tuple[int, int, int, int]) -> None:
        sx, sy, sw, sh = source_bbox
        if source.shape[:2] != (sh, sw):
            source = cv2.resize(source, (sw, sh), interpolation=cv2.INTER_NEAREST)

        overlap_x1 = max(bx, sx)
        overlap_y1 = max(by, sy)
        overlap_x2 = min(bx + bw, sx + sw)
        overlap_y2 = min(by + bh, sy + sh)
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return

        src_x1 = overlap_x1 - sx
        src_y1 = overlap_y1 - sy
        src_x2 = src_x1 + (overlap_x2 - overlap_x1)
        src_y2 = src_y1 + (overlap_y2 - overlap_y1)
        dst_x1 = overlap_x1 - bx
        dst_y1 = overlap_y1 - by
        dst_x2 = dst_x1 + (overlap_x2 - overlap_x1)
        dst_y2 = dst_y1 + (overlap_y2 - overlap_y1)
        crop = source[src_y1:src_y2, src_x1:src_x2]
        if crop.shape[:2] == (dst_y2 - dst_y1, dst_x2 - dst_x1):
            local[dst_y1:dst_y2, dst_x1:dst_x2] = cv2.bitwise_or(
                local[dst_y1:dst_y2, dst_x1:dst_x2],
                crop,
            )

    if group.mask is not None and group.mask.size > 0 and np.any(group.mask):
        if group.mask.shape[:2] == (group.h, group.w):
            paste(group.mask, group.bbox)
        elif group.mask.shape[0] >= by + bh and group.mask.shape[1] >= bx + bw:
            full_crop = group.mask[by : by + bh, bx : bx + bw]
            if full_crop.shape[:2] == (bh, bw):
                local = cv2.bitwise_or(local, full_crop)

    for rid in group.region_ids:
        region = regions_by_id.get(rid)
        if region is None:
            continue
        if (
            region.local_mask is None
            or region.local_mask.size == 0
            or not np.any(region.local_mask)
        ):
            continue
        paste(region.local_mask, region.bbox)
    return local


def _raw_text_bbox(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape
    mask = _build_group_local_mask(group, regions_by_id, group.bbox)
    if np.any(mask):
        points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if points is not None:
            mx, my, mw, mh = [int(value) for value in cv2.boundingRect(points)]
            x, y, w, h = group.x + mx, group.y + my, mw, mh
        else:
            x, y, w, h = group.bbox
    else:
        x, y, w, h = group.bbox
    x = max(0, min(x, max(0, img_w - 1)))
    y = max(0, min(y, max(0, img_h - 1)))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return (x, y, w, h)


def _tight_layout_bbox(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    cfg: TypesettingConfig,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Backward-compatible tight mask box used by tests and ``layout_mode=tight``."""

    img_h, img_w = image_shape
    x, y, w, h = group.bbox
    if cfg.layout_from_mask:
        mask = _build_group_local_mask(group, regions_by_id, group.bbox)
        if np.any(mask):
            radius = max(0, int(cfg.layout_mask_dilate))
            if radius > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (radius * 2 + 1, radius * 2 + 1),
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
            points = cv2.findNonZero((mask > 0).astype(np.uint8))
            if points is not None:
                mx, my, mw, mh = [int(value) for value in cv2.boundingRect(points)]
                pad = max(0, int(cfg.layout_padding_px))
                x = x + mx - pad
                y = y + my - pad
                w = mw + pad * 2
                h = mh + pad * 2

    x = max(0, min(x, max(0, img_w - 1)))
    y = max(0, min(y, max(0, img_h - 1)))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return (x, y, w, h)


def _mask_font_size_hint(mask: np.ndarray, direction: str) -> float | None:
    """Estimate glyph width/height from the original text-pixel projection.

    Detector ``font_size`` is useful but occasionally reports the full caption
    height (for example 130 px for a 75 px horizontal title).  The projection
    width of each vertical column, or projection height of each horizontal line,
    gives an independent estimate that can reject those extreme hints.
    """
    if mask.size == 0 or not np.any(mask):
        return None
    binary = (mask > 0).astype(np.uint8)
    profile = np.count_nonzero(binary, axis=0 if direction == "vertical" else 1).astype(float)
    if profile.size == 0 or float(profile.max()) <= 0:
        return None

    window = max(3, min(11, round(profile.size * 0.012)))
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(profile, np.ones(window, dtype=float), mode="same")
    active = (smooth >= max(1.0, float(smooth.max()) * 0.07)).astype(np.uint8)
    close = max(1, round(profile.size * 0.006))
    active = cv2.morphologyEx(
        active[None, :] * 255,
        cv2.MORPH_CLOSE,
        np.ones((1, close * 2 + 1), dtype=np.uint8),
    )[0] > 0

    widths: list[int] = []
    run_start: int | None = None
    for index, enabled in enumerate(list(active) + [False]):
        if enabled and run_start is None:
            run_start = index
        elif not enabled and run_start is not None:
            width = index - run_start
            if width >= 3:
                widths.append(width)
            run_start = None
    if not widths or len(widths) > 24:
        return None
    return float(np.median(widths))


def _preferred_group_font_size(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> float | None:
    regions = [regions_by_id[rid] for rid in group.region_ids if rid in regions_by_id]
    primary = [
        region.font_size_hint
        for region in regions
        if region.source == "ctd" and 4 <= region.font_size_hint <= 300
    ]
    hints = primary or [
        region.font_size_hint for region in regions if 4 <= region.font_size_hint <= 300
    ]
    detector_hint = float(np.median(hints)) if hints else None

    local_mask = _build_group_local_mask(group, regions_by_id, group.bbox)
    direction = "vertical" if group.vertical else "horizontal"
    mask_hint = _mask_font_size_hint(local_mask, direction)

    if detector_hint is None:
        preferred = mask_hint
    elif mask_hint is None:
        preferred = detector_hint
    else:
        ratio = detector_hint / max(1.0, mask_hint)
        if 0.62 <= ratio <= 1.55:
            preferred = detector_hint * 0.72 + mask_hint * 0.28
        else:
            preferred = mask_hint

    if preferred is None or preferred <= 0:
        return None
    return float(np.clip(preferred, 4.0, 300.0))


def _runs_from_projection(
    projection: np.ndarray,
    cross_size: int,
    preferred: float,
) -> list[tuple[int, int, float]]:
    if projection.size == 0 or not np.any(projection):
        return []
    threshold = max(2.0, cross_size * 0.003)
    active = (projection >= threshold).astype(np.uint8)[None, :] * 255
    kernel_size = max(1, round(preferred * 0.22))
    if kernel_size % 2 == 0:
        kernel_size += 1
    active = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        np.ones((1, kernel_size), dtype=np.uint8),
    )[0]

    runs: list[tuple[int, int, float]] = []
    start: int | None = None
    for index, enabled in enumerate(list(active > 0) + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            end = index
            width = end - start
            pixels = float(projection[start:end].sum())
            if width >= max(2, preferred * 0.16) and pixels >= preferred * preferred * 0.12:
                runs.append((start, end, pixels))
            start = None

    merged: list[tuple[int, int, float]] = []
    max_gap = preferred * 0.34
    for run in runs:
        if merged and run[0] - merged[-1][1] <= max_gap:
            previous = merged[-1]
            merged[-1] = (previous[0], run[1], previous[2] + run[2])
        else:
            merged.append(run)
    return merged


def _projection_centers(mask: np.ndarray, direction: str, preferred: float) -> list[float]:
    if mask.size == 0 or not np.any(mask):
        return []
    if direction == "vertical":
        projection = np.count_nonzero(mask > 0, axis=0).astype(np.float32)
        runs = _runs_from_projection(projection, mask.shape[0], preferred)
    else:
        projection = np.count_nonzero(mask > 0, axis=1).astype(np.float32)
        runs = _runs_from_projection(projection, mask.shape[1], preferred)

    centers: list[float] = []
    for start, end, _pixels in runs:
        weights = projection[start:end]
        indices = np.arange(start, end, dtype=np.float32)
        if float(weights.sum()) > 0:
            centers.append(float(np.average(indices, weights=weights)))
        else:
            centers.append((start + end - 1) / 2.0)
    return centers if len(centers) <= 20 else []


def _region_axis_centers(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    direction: str,
    preferred: float,
) -> list[float]:
    regions = [regions_by_id[rid] for rid in group.region_ids if rid in regions_by_id]
    primary = [region for region in regions if region.source == "ctd"] or regions
    values: list[float] = []
    for region in primary:
        if direction == "vertical":
            if region.w > preferred * 1.85:
                continue
            value = region.x + region.w / 2.0 - group.x
        else:
            if region.h > preferred * 1.85:
                continue
            value = region.y + region.h / 2.0 - group.y
        values.append(float(value))

    values.sort()
    clustered: list[list[float]] = []
    for value in values:
        if clustered and value - float(np.mean(clustered[-1])) <= preferred * 0.55:
            clustered[-1].append(value)
        else:
            clustered.append([value])
    return [float(np.mean(cluster)) for cluster in clustered]


def _infer_original_geometry(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    image_shape: tuple[int, int],
    direction: str,
) -> OriginalTextGeometry:
    raw_bbox = _raw_text_bbox(group, regions_by_id, image_shape)
    preferred = _preferred_group_font_size(group, regions_by_id)
    if preferred is None or preferred <= 0:
        if direction == "vertical":
            preferred = max(8.0, min(float(raw_bbox[2]), float(raw_bbox[3]) / 3.0))
        else:
            preferred = max(8.0, min(float(raw_bbox[3]), float(raw_bbox[2]) / 3.0))

    local_mask = _build_group_local_mask(group, regions_by_id, group.bbox)
    mask_centers = _projection_centers(local_mask, direction, preferred)
    region_centers = _region_axis_centers(group, regions_by_id, direction, preferred)
    centers = mask_centers or region_centers
    if region_centers and len(region_centers) > len(centers):
        centers = region_centers
    primary_count = max(1, len(centers))

    if len(centers) > 1:
        sorted_centers = sorted(centers)
        steps = [right - left for left, right in itertools.pairwise(sorted_centers)]
        primary_step = float(np.median(steps))
    else:
        primary_step = preferred * 1.08

    source_length = max(1, _visible_length(group.ocr_text))
    max_items = max(1, math.ceil(source_length / primary_count))
    target_extent = raw_bbox[3] if direction == "vertical" else raw_bbox[2]
    if max_items > 1:
        secondary_step = (target_extent - preferred) / (max_items - 1)
    else:
        secondary_step = preferred * 1.08
    secondary_step = float(
        np.clip(secondary_step, preferred * 0.82, preferred * 1.62)
    )

    return OriginalTextGeometry(
        bbox=raw_bbox,
        font_size=float(preferred),
        primary_count=primary_count,
        primary_step=max(preferred * 0.8, primary_step),
        secondary_step=secondary_step,
        source_length=source_length,
    )


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = image_shape
    x, y, w, h = bbox
    x = max(0, min(round(x), max(0, image_w - 1)))
    y = max(0, min(round(y), max(0, image_h - 1)))
    x2 = max(x + 1, min(round(x + w), image_w))
    y2 = max(y + 1, min(round(y + h), image_h))
    return (x, y, x2 - x, y2 - y)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    amount: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return _clip_bbox((x - amount, y - amount, w + amount * 2, h + amount * 2), image_shape)


def _bbox_union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _paste_mask_into_roi(
    roi_mask: np.ndarray,
    source: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    roi_bbox: tuple[int, int, int, int],
) -> None:
    sx, sy, sw, sh = source_bbox
    rx, ry, rw, rh = roi_bbox
    if source.shape[:2] != (sh, sw):
        source = cv2.resize(source, (sw, sh), interpolation=cv2.INTER_NEAREST)
    x1, y1 = max(sx, rx), max(sy, ry)
    x2, y2 = min(sx + sw, rx + rw), min(sy + sh, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return
    src = source[y1 - sy : y2 - sy, x1 - sx : x2 - sx]
    roi_mask[y1 - ry : y2 - ry, x1 - rx : x2 - rx] = cv2.bitwise_or(
        roi_mask[y1 - ry : y2 - ry, x1 - rx : x2 - rx],
        src,
    )


def _safe_background_bbox(
    image: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    preferred: float,
    cfg: TypesettingConfig,
) -> tuple[int, int, int, int] | None:
    """Derive a candidate bbox from protected-edge-aware safe-region evidence."""

    if not cfg.adaptive_bubble_layout:
        return None
    image_h, image_w = image.shape[:2]
    seed_x, seed_y, seed_w, seed_h = seed_bbox
    expand = min(
        cfg.bubble_search_max_px,
        max(
            round(max(seed_w, seed_h) * cfg.bubble_search_expand_ratio),
            round(preferred * 1.5),
        ),
    )
    search = _clip_bbox(
        (seed_x - expand, seed_y - expand, seed_w + expand * 2, seed_h + expand * 2),
        (image_h, image_w),
    )
    rx, ry, rw, rh = search
    roi = image[ry : ry + rh, rx : rx + rw]
    if roi.size == 0:
        return None

    text_mask = np.zeros((rh, rw), dtype=np.uint8)
    group_mask = _build_group_local_mask(group, regions_by_id, group.bbox)
    if np.any(group_mask):
        _paste_mask_into_roi(text_mask, group_mask, group.bbox, search)
    local_polygons: list[tuple[tuple[float, float], ...]] = []
    for region_id in group.region_ids:
        region = regions_by_id.get(region_id)
        if region is None:
            continue
        local_polygons.extend(
            tuple((px - rx, py - ry) for px, py in polygon)
            for polygon in region.line_polygons
        )
    artifacts = build_safe_region(roi, text_mask, line_polygons=tuple(local_polygons))
    points = cv2.findNonZero((artifacts.render_mask > 0).astype(np.uint8))
    if points is None:
        return None
    x1, y1, width_value, height_value = cv2.boundingRect(points)
    candidate = (rx + x1, ry + y1, width_value, height_value)
    if candidate[2] <= seed_w and candidate[3] <= seed_h:
        return None

    margin = max(1, round(min(candidate[2], candidate[3]) * cfg.bubble_inner_margin_ratio))
    cx1 = min(seed_x, candidate[0] + margin)
    cy1 = min(seed_y, candidate[1] + margin)
    cx2 = max(seed_x + seed_w, candidate[0] + candidate[2] - margin)
    cy2 = max(seed_y + seed_h, candidate[1] + candidate[3] - margin)
    return _clip_bbox((cx1, cy1, cx2 - cx1, cy2 - cy1), (image_h, image_w))


def _select_layout_bbox(
    image: np.ndarray,
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    geometry: OriginalTextGeometry,
    cfg: TypesettingConfig,
) -> tuple[int, int, int, int]:
    image_shape = image.shape[:2]
    if cfg.render_scope == "region" and group.region_ids:
        first = regions_by_id.get(group.region_ids[0])
        return _clip_bbox(first.bbox if first is not None else group.bbox, image_shape)
    if cfg.render_scope == "group_bbox" or cfg.layout_mode == "group":
        return _clip_bbox(group.bbox, image_shape)
    if cfg.layout_mode == "tight":
        return _tight_layout_bbox(group, regions_by_id, cfg, image_shape)

    tight = geometry.bbox
    tight_area = max(1, tight[2] * tight[3])
    group_area = max(1, group.w * group.h)
    reasonable_group = (
        group_area / tight_area <= 2.35
        and group.w <= tight[2] + geometry.font_size * 2.6
        and group.h <= tight[3] + geometry.font_size * 2.6
    )
    base = _bbox_union(tight, group.bbox) if reasonable_group else tight
    dynamic_padding = max(
        cfg.layout_padding_px,
        round(geometry.font_size * cfg.layout_padding_ratio),
    )
    base = _expand_bbox(base, dynamic_padding, image_shape)
    bubble = _safe_background_bbox(
        image,
        base,
        group,
        regions_by_id,
        geometry.font_size,
        cfg,
    )
    return bubble if bubble is not None else base


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(value, upper))


def _safe_log_ratio(value: float, target: float) -> float:
    return abs(math.log(max(value, 1e-6) / max(target, 1e-6)))


def _choose_layout_candidate(
    candidates: list[tuple[TextLayoutPlan, float]],
    preferred_font_size: float,
    cfg: TypesettingConfig,
) -> TextLayoutPlan | None:
    """Choose geometry without sacrificing a readable original-size font.

    The former layout path mechanically kept decreasing font size until the text
    fit a rectangle.  Even a perfectly viable original-size plan could lose to a
    smaller plan merely because the smaller block matched the target extent a bit
    better.  This two-stage choice first locks onto the original-size band; only
    when *no* candidate in that band fits may it use the limited shrink band.
    """

    if not candidates:
        return None
    preserve_floor = max(
        float(cfg.font_size_min),
        preferred_font_size * cfg.font_preserve_floor_scale,
    )
    near_original = [
        item for item in candidates if item[0].font_size + 1e-6 >= preserve_floor
    ]
    pool = near_original or candidates
    return min(
        pool,
        key=lambda item: (
            item[1],
            abs(math.log(max(item[0].font_size, 1) / max(preferred_font_size, 1e-6))),
            -item[0].font_size,
        ),
    )[0]


def layout_plan_block_bbox(plan: TextLayoutPlan) -> tuple[int, int, int, int]:
    """Return the actual planned text block in full-image coordinates."""

    x = plan.bbox[0] + plan.center_x - plan.block_width / 2.0
    y = plan.bbox[1] + plan.center_y - plan.block_height / 2.0
    x1 = math.floor(x)
    y1 = math.floor(y)
    x2 = math.ceil(x + plan.block_width)
    y2 = math.ceil(y + plan.block_height)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _plan_vertical(
    text: str,
    layout_bbox: tuple[int, int, int, int],
    geometry: OriginalTextGeometry,
    font_path: str,
    cfg: TypesettingConfig,
) -> TextLayoutPlan:
    _x, _y, width, height = layout_bbox
    pad = cfg.inner_padding
    available_w = max(1.0, width - pad * 2.0)
    available_h = max(1.0, height - pad * 2.0)
    preferred = geometry.font_size * cfg.font_size_scale
    low, high = _font_bounds(cfg, geometry.font_size)
    target_x = geometry.bbox[0] + geometry.bbox[2] / 2.0 - layout_bbox[0]
    target_y = geometry.bbox[1] + geometry.bbox[3] / 2.0 - layout_bbox[1]
    target_w = max(preferred, float(geometry.bbox[2]))
    target_h = max(preferred, float(geometry.bbox[3]))
    text_length = len(grapheme_clusters(text))
    length_ratio = text_length / max(1, geometry.source_length)
    expected_columns = max(
        1,
        round(geometry.primary_count * math.sqrt(max(0.15, length_ratio))),
    )

    candidates: list[tuple[TextLayoutPlan, float]] = []
    for size in range(high, low - 1, -1):
        font = _load_font(font_path, size)
        _ascent, _descent, line_h = _get_font_metrics(font)
        glyph_bbox = font.getbbox("國")
        glyph_w = max(1.0, float(glyph_bbox[2] - glyph_bbox[0]))
        glyph_h = max(1.0, float(glyph_bbox[3] - glyph_bbox[1]))
        base_char_step = max(glyph_h, line_h * cfg.vertical_char_spacing)
        min_char_step = base_char_step * cfg.min_char_spacing_ratio
        max_char_step = base_char_step * cfg.max_char_spacing_ratio
        min_col_step = max(glyph_w, size * cfg.min_column_spacing_ratio)
        max_col_step = size * cfg.max_column_spacing_ratio
        max_columns = min(
            text_length,
            max(1, math.floor((available_w - glyph_w) / max(1.0, min_col_step)) + 1),
        )

        for columns in range(1, max_columns + 1):
            chunks = balanced_legal_chunks(text, columns)
            max_items = max(len(grapheme_clusters(chunk)) for chunk in chunks)
            if columns > 1 and text_length / columns < cfg.min_chars_per_column * 0.65:
                sparse_penalty = (cfg.min_chars_per_column - text_length / columns) * 0.7
            else:
                sparse_penalty = 0.0

            if max_items <= 1:
                char_step = base_char_step
                block_h = glyph_h
            else:
                max_fit_char_step = (available_h - glyph_h) / (max_items - 1)
                if max_fit_char_step + 1e-6 < min_char_step:
                    continue
                scaled_original = geometry.secondary_step * size / max(geometry.font_size, 1.0)
                fill_step = (target_h - glyph_h) / (max_items - 1)
                desired = scaled_original
                if fill_step > desired:
                    desired = min(fill_step, max_char_step)
                else:
                    desired = max(fill_step, min_char_step)
                char_step = _clamp(
                    desired,
                    min_char_step,
                    min(max_char_step, max_fit_char_step),
                )
                block_h = glyph_h + (max_items - 1) * char_step

            if columns <= 1:
                col_step = 0.0
                block_w = glyph_w
            else:
                max_fit_col_step = (available_w - glyph_w) / (columns - 1)
                if max_fit_col_step + 1e-6 < min_col_step:
                    continue
                scaled_original = geometry.primary_step * size / max(geometry.font_size, 1.0)
                fill_step = (target_w - glyph_w) / (columns - 1)
                if columns < geometry.primary_count and fill_step > scaled_original:
                    desired = min(fill_step, scaled_original * 1.22)
                elif columns > geometry.primary_count:
                    desired = min(scaled_original, fill_step)
                else:
                    desired = scaled_original
                col_step = _clamp(
                    desired,
                    min_col_step,
                    min(max_col_step, max_fit_col_step),
                )
                block_w = glyph_w + (columns - 1) * col_step

            if block_w > available_w + 1e-6 or block_h > available_h + 1e-6:
                continue

            font_cost = 4.4 * _safe_log_ratio(size, preferred)
            extent_cost = (
                1.15 * _safe_log_ratio(block_w, target_w)
                + 1.45 * _safe_log_ratio(block_h, target_h)
            )
            column_cost = 0.32 * abs(columns - expected_columns)
            original_char = max(1.0, geometry.secondary_step * size / max(geometry.font_size, 1.0))
            original_col = max(1.0, geometry.primary_step * size / max(geometry.font_size, 1.0))
            spacing_cost = 0.18 * _safe_log_ratio(char_step, original_char)
            if columns > 1:
                spacing_cost += 0.14 * _safe_log_ratio(col_step, original_col)
            cost = font_cost + extent_cost + column_cost + spacing_cost + sparse_penalty

            center_x = _clamp(
                target_x,
                pad + block_w / 2.0,
                width - pad - block_w / 2.0,
            )
            center_y = _clamp(
                target_y,
                pad + block_h / 2.0,
                height - pad - block_h / 2.0,
            )
            plan = TextLayoutPlan(
                bbox=layout_bbox,
                direction="vertical",
                font_size=size,
                chunks=chunks,
                primary_step=col_step,
                secondary_step=char_step,
                center_x=center_x,
                center_y=center_y,
                block_width=block_w,
                block_height=block_h,
                score=-cost,
            )
            candidates.append((plan, cost))

    best = _choose_layout_candidate(candidates, preferred, cfg)
    if best is not None:
        return best
    return TextLayoutPlan(
        bbox=layout_bbox,
        direction="vertical",
        font_size=low,
        chunks=(),
        primary_step=0.0,
        secondary_step=0.0,
        center_x=target_x,
        center_y=target_y,
        block_width=0.0,
        block_height=0.0,
        fits=False,
        reason=f"translation_requires_font_below_{low}px",
    )


def _measure_line(
    line: str,
    size: int,
    primary_font: str,
    fallback_font: str | None,
    replace_unsupported: bool,
) -> float:
    total = 0.0
    for char in line:
        font, draw_char = _get_font_and_char(
            char,
            size,
            primary_font,
            fallback_font,
            replace_unsupported,
        )
        total += _measure_char_advance(font, draw_char)
    return total


def _plan_horizontal(
    text: str,
    layout_bbox: tuple[int, int, int, int],
    geometry: OriginalTextGeometry,
    primary_font: str,
    fallback_font: str | None,
    cfg: TypesettingConfig,
) -> TextLayoutPlan:
    _x, _y, width, height = layout_bbox
    pad = cfg.inner_padding
    available_w = max(1.0, width - pad * 2.0)
    available_h = max(1.0, height - pad * 2.0)
    preferred = geometry.font_size * cfg.font_size_scale
    low, high = _font_bounds(cfg, geometry.font_size)
    target_x = geometry.bbox[0] + geometry.bbox[2] / 2.0 - layout_bbox[0]
    target_y = geometry.bbox[1] + geometry.bbox[3] / 2.0 - layout_bbox[1]
    target_w = max(preferred, float(geometry.bbox[2]))
    target_h = max(preferred, float(geometry.bbox[3]))

    candidates: list[tuple[TextLayoutPlan, float]] = []
    for size in range(high, low - 1, -1):
        font = _load_font(primary_font, size)
        _ascent, _descent, line_h = _get_font_metrics(font)
        lines = _wrap_horizontal(text, font, int(available_w))
        if not lines:
            continue

        raw_widths = [
            _measure_line(
                line,
                size,
                primary_font,
                fallback_font,
                cfg.replace_unsupported_glyphs,
            )
            for line in lines
        ]
        longest_gaps = max((max(0, len(line) - 1) for line in lines), default=0)
        max_tracking = size * max(0.0, cfg.max_char_spacing_ratio - 1.0)
        desired_tracking = 0.0
        if longest_gaps > 0:
            raw_block_w = max(raw_widths, default=0.0)
            desired_tracking = max(0.0, (target_w - raw_block_w) / longest_gaps)
        tracking = min(max_tracking, desired_tracking)
        line_widths = [
            width_value + max(0, len(line) - 1) * tracking
            for line, width_value in zip(lines, raw_widths)
        ]
        block_w = max(line_widths, default=0.0)

        base_line_step = max(1.0, line_h * cfg.line_spacing)
        if len(lines) <= 1:
            line_step = 0.0
            block_h = float(line_h)
        else:
            max_fit_step = (available_h - line_h) / (len(lines) - 1)
            if max_fit_step < line_h * 0.82:
                continue
            fill_step = (target_h - line_h) / (len(lines) - 1)
            line_step = _clamp(
                max(base_line_step, fill_step),
                line_h * 0.88,
                min(line_h * 1.38, max_fit_step),
            )
            block_h = line_h + (len(lines) - 1) * line_step

        if block_w > available_w + 1e-6 or block_h > available_h + 1e-6:
            continue
        cost = (
            4.8 * _safe_log_ratio(size, preferred)
            + 1.55 * _safe_log_ratio(block_w, target_w)
            + 1.75 * _safe_log_ratio(block_h, target_h)
        )
        if tracking > 0:
            cost += 0.08 * _safe_log_ratio(
                max(1.0, tracking + size),
                max(1.0, size),
            )
        center_x = _clamp(target_x, pad + block_w / 2.0, width - pad - block_w / 2.0)
        center_y = _clamp(target_y, pad + block_h / 2.0, height - pad - block_h / 2.0)
        plan = TextLayoutPlan(
            bbox=layout_bbox,
            direction="horizontal",
            font_size=size,
            chunks=tuple(lines),
            primary_step=line_step,
            secondary_step=tracking,
            center_x=center_x,
            center_y=center_y,
            block_width=block_w,
            block_height=block_h,
            score=-cost,
        )
        candidates.append((plan, cost))

    best = _choose_layout_candidate(candidates, preferred, cfg)
    if best is not None:
        return best
    return TextLayoutPlan(
        bbox=layout_bbox,
        direction="horizontal",
        font_size=low,
        chunks=(),
        primary_step=0.0,
        secondary_step=0.0,
        center_x=target_x,
        center_y=target_y,
        block_width=0.0,
        block_height=0.0,
        fits=False,
        reason=f"translation_requires_font_below_{low}px",
    )


def plan_text_layout(
    image: np.ndarray,
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    text: str,
    font_path: str | Path,
    cfg: TypesettingConfig | None = None,
    fallback_font_path: str | Path | None = None,
) -> TextLayoutPlan:
    """Plan once before inpainting so an unreadable result can retain its original."""

    cfg = cfg or TypesettingConfig()
    cleaned = _sanitize_render_text(text)
    direction = _decide_direction(group, cfg.direction)
    if direction == "vertical":
        cleaned = cleaned.replace(" ", "")
    if not cleaned:
        return TextLayoutPlan(
            bbox=group.bbox,
            direction=direction,
            font_size=cfg.font_size_min,
            chunks=(),
            primary_step=0.0,
            secondary_step=0.0,
            center_x=0.0,
            center_y=0.0,
            block_width=0.0,
            block_height=0.0,
            fits=False,
            reason="empty_translation",
        )

    primary_font = str(Path(font_path).resolve())
    fallback_font: str | None = None
    if fallback_font_path is not None:
        candidate = Path(fallback_font_path).resolve()
        if candidate.exists():
            fallback_font = str(candidate)

    geometry = _infer_original_geometry(group, regions_by_id, image.shape[:2], direction)
    layout_bbox = _select_layout_bbox(image, group, regions_by_id, geometry, cfg)
    if direction == "vertical":
        return _plan_vertical(cleaned, layout_bbox, geometry, primary_font, cfg)
    return _plan_horizontal(cleaned, layout_bbox, geometry, primary_font, fallback_font, cfg)


def _simple_patch_plan(
    patch_shape: tuple[int, int],
    text: str,
    direction: str,
    font_path: str,
    fallback_font: str | None,
    cfg: TypesettingConfig,
    preferred_font_size: float | None,
) -> TextLayoutPlan:
    height, width = patch_shape
    preferred = preferred_font_size or min(width, height) * 0.42
    geometry = OriginalTextGeometry(
        bbox=(cfg.inner_padding, cfg.inner_padding, max(1, width - cfg.inner_padding * 2), max(1, height - cfg.inner_padding * 2)),
        font_size=max(float(cfg.font_size_min), float(preferred)),
        primary_count=1,
        primary_step=float(preferred),
        secondary_step=float(preferred) * 1.08,
        source_length=max(1, len(grapheme_clusters(text))),
    )
    bbox = (0, 0, width, height)
    if direction == "vertical":
        return _plan_vertical(text, bbox, geometry, font_path, cfg)
    return _plan_horizontal(text, bbox, geometry, font_path, fallback_font, cfg)


def _detect_text_color_from_patch(patch_bgr: np.ndarray) -> tuple[int, int, int]:
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return (255, 255, 255) if float(gray.mean()) < 128 else (0, 0, 0)


def _maybe_outline_draw(
    draw: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    cfg: TypesettingConfig,
) -> None:
    ow = max(0, int(cfg.outline_width))
    if ow <= 0:
        draw.text(pos, text, font=font, fill=fill)
        return

    if cfg.outline_color == "auto":
        outline_rgb = (255, 255, 255) if fill[:3] == (0, 0, 0) else (0, 0, 0)
    elif isinstance(cfg.outline_color, str):
        outline_rgb = (0, 0, 0)
    else:
        outline_rgb = tuple(int(value) for value in cfg.outline_color)
    draw.text(
        pos,
        text,
        font=font,
        fill=fill,
        stroke_width=ow,
        stroke_fill=(*outline_rgb, 255),
    )


def render_text_into_patch(
    patch_bgr: np.ndarray,
    text: str,
    direction: str,
    font_path: str | Path,
    cfg: TypesettingConfig,
    fallback_font_path: str | Path | None = None,
    clip_mask: np.ndarray | None = None,
    preferred_font_size: float | None = None,
    layout_plan: TextLayoutPlan | None = None,
) -> np.ndarray:
    """Render a preflighted plan into an RGBA patch."""

    height, width = patch_bgr.shape[:2]
    cleaned = _sanitize_render_text(text)
    if direction == "vertical":
        cleaned = cleaned.replace(" ", "")
    if height <= 1 or width <= 1 or not cleaned:
        return np.zeros((height, width, 4), dtype=np.uint8)

    primary_font = str(Path(font_path).resolve())
    fallback_font: str | None = None
    if fallback_font_path is not None:
        candidate = Path(fallback_font_path).resolve()
        if candidate.exists():
            fallback_font = str(candidate)

    plan = layout_plan or _simple_patch_plan(
        (height, width),
        cleaned,
        direction,
        primary_font,
        fallback_font,
        cfg,
        preferred_font_size,
    )
    if not plan.fits or not plan.chunks:
        return np.zeros((height, width, 4), dtype=np.uint8)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color_rgb = (
        _detect_text_color_from_patch(patch_bgr)
        if cfg.text_color == "auto"
        else (
            (0, 0, 0)
            if isinstance(cfg.text_color, str)
            else tuple(int(value) for value in cfg.text_color)
        )
    )
    fill = (*color_rgb, 255)
    size = plan.font_size

    if plan.direction == "horizontal":
        top = plan.center_y - plan.block_height / 2.0
        for line_index, line in enumerate(plan.chunks):
            line_width = _measure_line(
                line,
                size,
                primary_font,
                fallback_font,
                cfg.replace_unsupported_glyphs,
            ) + max(0, len(line) - 1) * plan.secondary_step
            cursor_x = plan.center_x - line_width / 2.0
            y_cell = top + line_index * plan.primary_step
            for char_index, char in enumerate(line):
                font, draw_char = _get_font_and_char(
                    char,
                    size,
                    primary_font,
                    fallback_font,
                    cfg.replace_unsupported_glyphs,
                )
                bbox = font.getbbox(draw_char)
                _maybe_outline_draw(
                    draw,
                    (cursor_x - bbox[0], y_cell - bbox[1]),
                    draw_char,
                    font,
                    fill,
                    cfg,
                )
                cursor_x += _measure_char_advance(font, draw_char)
                if char_index + 1 < len(line):
                    cursor_x += plan.secondary_step
    else:
        base_font = _load_font(primary_font, size)
        glyph_bbox = base_font.getbbox("國")
        cell_width = max(1.0, float(glyph_bbox[2] - glyph_bbox[0]))
        cell_height = max(1.0, float(glyph_bbox[3] - glyph_bbox[1]))
        left = plan.center_x - plan.block_width / 2.0
        column_count = len(plan.chunks)
        for column_index, column in enumerate(plan.chunks):
            column_left = left + (column_count - 1 - column_index) * plan.primary_step
            column_height = cell_height if len(column) <= 1 else (
                cell_height + (len(column) - 1) * plan.secondary_step
            )
            top = plan.center_y - column_height / 2.0
            for char_index, char in enumerate(column):
                font, draw_char = _get_font_and_char(
                    char,
                    size,
                    primary_font,
                    fallback_font,
                    cfg.replace_unsupported_glyphs,
                )
                bbox = font.getbbox(draw_char)
                char_width = float(bbox[2] - bbox[0])
                x = column_left + (cell_width - char_width) / 2.0 - bbox[0]
                y = top + char_index * plan.secondary_step - bbox[1]
                _maybe_outline_draw(draw, (x, y), draw_char, font, fill, cfg)

    layer_np = np.array(layer)
    if clip_mask is not None:
        mask = clip_mask
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        alpha = layer_np[:, :, 3].astype(np.uint16)
        layer_np[:, :, 3] = (alpha * mask.astype(np.uint16) // 255).astype(np.uint8)
    return layer_np


def compose_patch_back(image: np.ndarray, patch_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-compose only the target ROI."""

    if patch_rgba.ndim != 3 or patch_rgba.shape[2] != 4:
        raise ValueError("patch_rgba 必須是 HxWx4")

    image_h, image_w = image.shape[:2]
    patch_h, patch_w = patch_rgba.shape[:2]
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(image_w, x + patch_w)
    dst_y2 = min(image_h, y + patch_h)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return image.copy()

    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    patch = patch_rgba[src_y1:src_y2, src_x1:src_x2]

    alpha = patch[:, :, 3:4].astype(np.uint16)
    foreground_bgr = patch[:, :, :3][:, :, ::-1].astype(np.uint16)
    result = image.copy()
    background = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.uint16)
    blended = (foreground_bgr * alpha + background * (255 - alpha) + 127) // 255
    result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
    return result


def render_text_into_group(
    image: np.ndarray,
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    text: str,
    font_path: str | Path,
    cfg: TypesettingConfig | None = None,
    fallback_font_path: str | Path | None = None,
    layout_plan: TextLayoutPlan | None = None,
    layout_reference_image: np.ndarray | None = None,
) -> np.ndarray:
    cfg = cfg or TypesettingConfig()
    reference = layout_reference_image if layout_reference_image is not None else image
    plan = layout_plan or plan_text_layout(
        reference,
        group,
        regions_by_id,
        text,
        font_path,
        cfg,
        fallback_font_path,
    )
    if not plan.fits:
        return image

    x, y, w, h = plan.bbox
    patch = image[y : y + h, x : x + w].copy()
    layer = render_text_into_patch(
        patch,
        text,
        direction=plan.direction,
        font_path=font_path,
        cfg=cfg,
        fallback_font_path=fallback_font_path,
        clip_mask=None,
        preferred_font_size=_preferred_group_font_size(group, regions_by_id),
        layout_plan=plan,
    )
    group.layout_bbox = plan.bbox
    group.rendered_font_size = int(plan.font_size)
    group.rendered_direction = plan.direction
    group.layout_mode = cfg.layout_mode
    group.layout_info = plan.to_dict()
    group.layout_info["block_bbox"] = {
        key: value
        for key, value in zip(("x", "y", "w", "h"), layout_plan_block_bbox(plan))
    }
    return compose_patch_back(image, layer, x, y)


def render_text_into_region(
    image: np.ndarray,
    region: TextRegion,
    text: str,
    font_path: str | Path,
    cfg: TypesettingConfig | None = None,
    fallback_font_path: str | Path | None = None,
) -> np.ndarray:
    """Backward-compatible region API."""

    pseudo_group = TextGroup(
        id=f"group_{region.id}",
        region_ids=[region.id],
        bbox=region.bbox,
        vertical=region.vertical,
    )
    return render_text_into_group(
        image=image,
        group=pseudo_group,
        regions_by_id={region.id: region},
        text=text,
        font_path=font_path,
        cfg=cfg,
        fallback_font_path=fallback_font_path,
    )
