"""文字擦除／修復模組：只處理已確認可渲染且具有可靠像素 mask 的群組。"""

from __future__ import annotations

import cv2
import numpy as np

from .config import InpaintingConfig
from .detector import DetectionResult, TextGroup


def _prepare_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return (mask > 0).astype(np.uint8) * 255


def _active_groups(
    groups: list[TextGroup],
    cfg: InpaintingConfig,
) -> list[TextGroup]:
    if not cfg.only_translated_groups:
        return groups
    return [
        group
        for group in groups
        if group.translation_valid and bool(group.translation.strip())
    ]


def _paste_local_mask(
    page_mask: np.ndarray,
    local_mask: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> bool:
    target_h, target_w = page_mask.shape[:2]
    if local_mask.shape[:2] != (h, w):
        local_mask = cv2.resize(local_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(target_w, x + w)
    y2 = min(target_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return False

    source_x1 = x1 - x
    source_y1 = y1 - y
    source_x2 = source_x1 + (x2 - x1)
    source_y2 = source_y1 + (y2 - y1)
    crop = local_mask[source_y1:source_y2, source_x1:source_x2]
    if crop.shape[:2] != (y2 - y1, x2 - x1) or not np.any(crop):
        return False

    page_mask[y1:y2, x1:x2] = cv2.bitwise_or(page_mask[y1:y2, x1:x2], crop)
    return True


def _build_group_union_mask(
    groups: list[TextGroup],
    detection_result: DetectionResult,
    target_h: int,
    target_w: int,
    *,
    allow_bbox_fallback: bool = False,
) -> np.ndarray:
    """建立精確文字像素 mask。

    過去在 mask 缺失時退回整個 region rectangle，會把人物臉部、頭髮與對話框
    背景一起送進 inpainting。現在預設直接略過該群組；只有明確開啟相容選項時
    才允許矩形退回。
    """
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    region_by_id = {region.id: region for region in detection_result.regions_post}

    for group in groups:
        pasted = False
        if group.mask is not None and group.mask.size > 0 and np.any(group.mask):
            if group.mask.shape[:2] == (target_h, target_w):
                # Only an exact page-sized array is allowed to use the legacy
                # full-page path.  Treating every unexpected shape as a page mask
                # can enlarge a tiny glyph mask across the entire image.
                group_mask = _prepare_mask(group.mask, target_h, target_w)
                mask = cv2.bitwise_or(mask, group_mask)
                pasted = bool(np.any(group_mask))
            else:
                # Local masks from older detector revisions may be a few pixels
                # different from group.bbox.  Resize them locally, never globally.
                pasted = _paste_local_mask(
                    mask,
                    group.mask,
                    x=group.x,
                    y=group.y,
                    w=group.w,
                    h=group.h,
                )

        if pasted:
            continue

        for region_id in group.region_ids:
            region = region_by_id.get(region_id)
            if region is None:
                continue
            local = region.local_mask
            if local is not None and local.size > 0 and np.any(local):
                pasted = _paste_local_mask(
                    mask,
                    local,
                    x=region.x,
                    y=region.y,
                    w=region.w,
                    h=region.h,
                ) or pasted
            elif allow_bbox_fallback:
                x1 = max(0, region.x)
                y1 = max(0, region.y)
                x2 = min(target_w, region.x + region.w)
                y2 = min(target_h, region.y + region.h)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
                    pasted = True

    return mask


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    return cv2.dilate(mask, kernel, iterations=1)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    return tuple(int(v) for v in cv2.boundingRect(points))


def _refine_flat_text_edge_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    cfg: InpaintingConfig,
) -> np.ndarray:
    """Recover antialiased glyph edges without expanding into nearby artwork.

    The refinement is used only after the surrounding ring has already been
    classified as a flat/dominant-color background.  Candidate pixels must be
    both close to the detector mask and visibly different from that background.
    This removes the pale Japanese outlines left by a core-only segmentation mask
    while avoiding the old failure mode of blindly dilating into faces or line art.
    """

    radius = max(0, int(cfg.hybrid_flat_edge_expand))
    if radius <= 0 or not np.any(mask):
        return mask

    zone = _dilate_mask(mask, radius) > 0
    image_float = image.astype(np.float32)
    difference = np.max(
        np.abs(image_float - background.reshape(1, 1, 3)),
        axis=2,
    )
    candidates = zone & (mask == 0) & (difference >= cfg.hybrid_flat_edge_contrast)
    if not np.any(candidates):
        return mask

    base_count = int(np.count_nonzero(mask))
    max_total = max(
        base_count,
        int(round(base_count * cfg.hybrid_flat_edge_max_growth)),
    )
    allowed_extra = max(0, max_total - base_count)
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count > allowed_extra:
        # Prefer pixels nearest to a confirmed text core, then the strongest
        # background contrast.  This keeps thin antialiasing halos but drops a
        # nearby panel border should it enter the search radius.
        outside = (mask == 0).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        ys, xs = np.nonzero(candidates)
        order = np.lexsort((-difference[ys, xs], distance[ys, xs]))
        selected = order[:allowed_extra]
        limited = np.zeros_like(candidates)
        limited[ys[selected], xs[selected]] = True
        candidates = limited

    refined = mask.copy()
    refined[candidates] = 255
    return refined


def _flat_background_refined_mask(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: InpaintingConfig,
) -> np.ndarray | None:
    """Return an edge-complete text mask when the local background is safe.

    A single median fill color leaves glyph-shaped patches on translucent or
    gently graded caption backgrounds.  The mask is therefore refined here, but
    the actual pixel reconstruction is performed by small-radius local inpainting.
    """
    # Sample the background *outside* the possible antialiasing halo.  Using a
    # ring immediately adjacent to a core-only detector mask lets gray glyph edges
    # poison the background statistics and incorrectly routes a white bubble to
    # Telea.
    edge_guard = max(0, int(cfg.hybrid_flat_edge_expand))
    ring_inner = _dilate_mask(mask, edge_guard)
    ring_outer = _dilate_mask(mask, edge_guard + cfg.hybrid_ring_radius)
    ring = cv2.bitwise_and(ring_outer, cv2.bitwise_not(ring_inner))
    pixels = image[ring > 0]
    if pixels.shape[0] < cfg.hybrid_min_ring_pixels:
        return None

    pixels_float = pixels.astype(np.float32)
    channel_std = np.std(pixels_float, axis=0)
    median = np.median(pixels_float, axis=0)
    distance = np.max(np.abs(pixels_float - median), axis=1)
    dominant_ratio = float(
        np.mean(distance <= float(cfg.hybrid_dominant_color_tolerance))
    )
    is_flat = float(np.max(channel_std)) <= cfg.hybrid_flat_std_threshold
    has_dominant_color = dominant_ratio >= cfg.hybrid_dominant_color_ratio
    if not (is_flat or has_dominant_color):
        return None

    return _refine_flat_text_edge_mask(image, mask, median, cfg)


def _inpaint_one_mask(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: InpaintingConfig,
    *,
    method: str,
) -> np.ndarray:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return image

    x, y, w, h = bbox
    margin = max(
        int(np.ceil(cfg.inpaint_radius * 3)),
        cfg.hybrid_ring_radius + 2,
        4,
    )
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2]
    local_mask = mask[y1:y2, x1:x2]
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    repaired = cv2.inpaint(
        roi,
        local_mask,
        inpaintRadius=float(cfg.inpaint_radius),
        flags=flag,
    )
    result = image.copy()
    result[y1:y2, x1:x2] = repaired
    return result


def _hybrid_inpaint(
    image: np.ndarray,
    groups: list[TextGroup],
    detection_result: DetectionResult,
    cfg: InpaintingConfig,
) -> np.ndarray:
    result = image.copy()
    target_h, target_w = image.shape[:2]
    total_dilate = max(0, cfg.mask_dilate) + max(0, cfg.extra_mask_dilate)

    # 逐群組處理，避免某個複雜背景讓整頁所有白底對話框都被迫走 Telea。
    for group in groups:
        mask = _build_group_union_mask(
            [group],
            detection_result,
            target_h,
            target_w,
            allow_bbox_fallback=cfg.allow_bbox_fallback,
        )
        mask = _dilate_mask(mask, total_dilate)
        if not np.any(mask):
            continue

        refined_mask = _flat_background_refined_mask(image, mask, cfg)
        if refined_mask is not None:
            result = _inpaint_one_mask(result, refined_mask, cfg, method="telea")
            continue

        result = _inpaint_one_mask(result, mask, cfg, method="telea")
    return result


def inpaint_regions(
    image: np.ndarray,
    detection_result: DetectionResult,
    cfg: InpaintingConfig | None = None,
) -> np.ndarray:
    cfg = cfg or InpaintingConfig()
    target_h, target_w = image.shape[:2]
    groups = _active_groups(detection_result.groups, cfg)

    if cfg.only_translated_groups and not groups:
        return image.copy()

    if cfg.method == "hybrid":
        return _hybrid_inpaint(image, groups, detection_result, cfg)

    if cfg.use_group_union_mask or cfg.only_translated_groups:
        mask = _build_group_union_mask(
            groups,
            detection_result,
            target_h,
            target_w,
            allow_bbox_fallback=cfg.allow_bbox_fallback,
        )
    else:
        mask = detection_result.mask
    mask = _prepare_mask(mask, target_h=target_h, target_w=target_w)

    if not np.any(mask):
        return image.copy()

    total_dilate = max(0, cfg.mask_dilate) + max(0, cfg.extra_mask_dilate)
    mask = _dilate_mask(mask, total_dilate)

    if cfg.method == "white":
        result = image.copy()
        result[mask > 0] = 255
        return result

    return _inpaint_one_mask(image, mask, cfg, method=cfg.method)
