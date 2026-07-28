"""主流水線：detect → mask fallback → OCR ensemble → fuzzy dedup → translate → safe render。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .artifacts import dump_debug_artifacts
from .config import AppConfig, PostprocessConfig
from .contracts.mapping import (
    MappingContractError,
    RequestMap,
    ValidatedTranslationBatch,
    bind_validated_responses,
    bind_validated_values,
    build_request_map,
    mapping_chain_template,
)
from .detector import DetectionResult, TextGroup, TextRegion, detect_text_regions
from .geometry import center_distance, containment_ratio, iom, merge_bbox
from .image_io import (
    ImageEncodeError,
    ImageWriteError,
    read_image,
    write_image,
    write_image_or_raise,
)
from .inpainter import inpaint_regions
from .ocr import (
    OCRInitializationError,
    assess_ocr_result,
    initialize_ocr_model,
    normalize_ocr_text,
    ocr_group_detailed,
)
from .profiling import profile_page, profile_span, set_page_profile_metrics
from .result import (
    BatchResult,
    GroupMappingSnapshot,
    PageResult,
    ResultIssue,
    derive_batch_status,
)
from .translator import (
    load_glossary,
    sanitize_translation_text,
    translate_batch_mapped,
    translate_page_mapped,
    translate_with_context_mapped,
    validate_translation,
)
from .typesetter import (
    TextLayoutPlan,
    layout_plan_block_bbox,
    plan_text_layout,
    render_text_into_group,
)

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_sort_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=_natural_sort_key)


def _group_area(group: TextGroup) -> int:
    return max(0, group.w) * max(0, group.h)


def _group_area_ratio(a: TextGroup, b: TextGroup) -> float:
    small = min(_group_area(a), _group_area(b))
    large = max(_group_area(a), _group_area(b), 1)
    return small / large


def _text_similarity(a: str, b: str) -> float:
    left = normalize_ocr_text(a, weak=True)
    right = normalize_ocr_text(b, weak=True)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _axis_overlap_ratio(a: TextGroup, b: TextGroup) -> float:
    """同一行／同一列的重疊程度；用來限制相鄰框去重。"""
    if a.vertical and b.vertical:
        overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return overlap / max(1, min(a.h, b.h))
    if not a.vertical and not b.vertical:
        overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        return overlap / max(1, min(a.w, b.w))
    return 0.0


def _bbox_gap_ratio(a: TextGroup, b: TextGroup) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    dx = max(0, max(b.x - ax2, a.x - bx2))
    dy = max(0, max(b.y - ay2, a.y - by2))
    gap = max(dx, dy)
    return gap / max(1, min(max(a.w, a.h), max(b.w, b.h)))


def _local_group_mask(group: TextGroup) -> np.ndarray | None:
    """回傳與 group.bbox 對齊的 mask，並相容舊版整頁 mask。"""
    if group.mask is None or group.mask.size == 0:
        return None
    if group.mask.shape[:2] == (group.h, group.w):
        return group.mask
    if group.mask.shape[0] >= group.y + group.h and group.mask.shape[1] >= group.x + group.w:
        return group.mask[group.y : group.y + group.h, group.x : group.x + group.w]
    return cv2.resize(group.mask, (group.w, group.h), interpolation=cv2.INTER_NEAREST)


def _paste_group_mask(
    canvas: np.ndarray,
    group: TextGroup,
    canvas_bbox: tuple[int, int, int, int],
) -> None:
    local = _local_group_mask(group)
    if local is None:
        return
    canvas_x, canvas_y, _canvas_w, _canvas_h = canvas_bbox
    x1 = group.x - canvas_x
    y1 = group.y - canvas_y
    x2 = x1 + group.w
    y2 = y1 + group.h
    if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
        return
    canvas[y1:y2, x1:x2] = cv2.bitwise_or(canvas[y1:y2, x1:x2], local)


def _group_mask_containment(inner: TextGroup, outer: TextGroup) -> float:
    """Return the fraction of ``inner`` text pixels also covered by ``outer``.

    Multi-scale detection often yields a whole-sentence mask plus one mask per
    vertical column.  Their mask IoU is small because the whole sentence is much
    larger, but nearly every pixel in the column is still contained by the outer
    mask.  IoU alone therefore misses exactly the overlap that later creates
    doubled subtitles.
    """
    inner_mask = _local_group_mask(inner)
    outer_mask = _local_group_mask(outer)
    if inner_mask is None or outer_mask is None:
        return 0.0

    inner_count = int(np.count_nonzero(inner_mask))
    if inner_count <= 0:
        return 0.0

    overlap_x1 = max(inner.x, outer.x)
    overlap_y1 = max(inner.y, outer.y)
    overlap_x2 = min(inner.x + inner.w, outer.x + outer.w)
    overlap_y2 = min(inner.y + inner.h, outer.y + outer.h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    inner_crop = inner_mask[
        overlap_y1 - inner.y : overlap_y2 - inner.y,
        overlap_x1 - inner.x : overlap_x2 - inner.x,
    ]
    outer_crop = outer_mask[
        overlap_y1 - outer.y : overlap_y2 - outer.y,
        overlap_x1 - outer.x : overlap_x2 - outer.x,
    ]
    if inner_crop.shape != outer_crop.shape or inner_crop.size == 0:
        return 0.0

    intersection = int(np.count_nonzero((inner_crop > 0) & (outer_crop > 0)))
    return intersection / inner_count


def _text_coverage(a: str, b: str) -> float:
    """How much of the shorter normalized string is explained by the longer."""
    left = normalize_ocr_text(a, weak=True)
    right = normalize_ocr_text(b, weak=True)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    shorter, longer = sorted((left, right), key=len)
    blocks = SequenceMatcher(None, shorter, longer, autojunk=False).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    return matched / max(1, len(shorter))


def _group_mask_iou(a: TextGroup, b: TextGroup) -> float:
    local_a = _local_group_mask(a)
    local_b = _local_group_mask(b)
    if local_a is None or local_b is None:
        return 0.0

    count_a = int(np.count_nonzero(local_a))
    count_b = int(np.count_nonzero(local_b))
    if count_a <= 0 or count_b <= 0:
        return 0.0

    overlap_x1 = max(a.x, b.x)
    overlap_y1 = max(a.y, b.y)
    overlap_x2 = min(a.x + a.w, b.x + b.w)
    overlap_y2 = min(a.y + a.h, b.y + b.h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    a_crop = local_a[
        overlap_y1 - a.y : overlap_y2 - a.y,
        overlap_x1 - a.x : overlap_x2 - a.x,
    ]
    b_crop = local_b[
        overlap_y1 - b.y : overlap_y2 - b.y,
        overlap_x1 - b.x : overlap_x2 - b.x,
    ]
    if a_crop.shape != b_crop.shape or a_crop.size == 0:
        return 0.0

    intersection = int(np.count_nonzero((a_crop > 0) & (b_crop > 0)))
    union = count_a + count_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _are_duplicate_groups(a: TextGroup, b: TextGroup, cfg: PostprocessConfig) -> bool:
    iom_score = iom(a, b)
    high_overlap = iom_score >= cfg.same_text_iom_thresh
    containment_ab = containment_ratio(a, b)
    containment_ba = containment_ratio(b, a)
    bbox_containment = max(containment_ab, containment_ba)
    contained = bbox_containment >= cfg.containment_ratio_thresh
    area_ratio = _group_area_ratio(a, b)
    max_dim = max(a.w, a.h, b.w, b.h, 1)
    center_ratio = center_distance(a, b) / max_dim
    mask_iou = _group_mask_iou(a, b)
    mask_containment_ab = _group_mask_containment(a, b)
    mask_containment_ba = _group_mask_containment(b, a)
    mask_containment = max(mask_containment_ab, mask_containment_ba)

    nested_geometry = (
        a.vertical == b.vertical
        and bbox_containment >= cfg.nested_fragment_containment
        and (mask_containment >= cfg.render_collision_mask_containment or center_ratio <= 0.82)
    )
    strong_geometry = (
        mask_iou >= 0.42
        or mask_containment >= cfg.render_collision_mask_containment
        or (high_overlap and area_ratio >= 0.30)
        or (contained and area_ratio >= 0.45 and center_ratio <= 0.70)
    )

    a_text = a.ocr_text_norm
    b_text = b.ocr_text_norm
    if strong_geometry and (not a_text or not b_text):
        # Empty OCR candidates are merged only when their actual text pixels overlap
        # strongly.  A giant empty panel bbox must not swallow real speech bubbles.
        return mask_iou >= 0.55 or (mask_containment >= 0.88 and area_ratio >= 0.18)
    if not a_text or not b_text:
        return False

    similarity = _text_similarity(a_text, b_text)
    coverage = _text_coverage(a_text, b_text)
    exact_or_substring = (
        a_text == b_text
        or (
            cfg.substring_match_enabled
            and (a_text in b_text or b_text in a_text)
            and min(len(a_text), len(b_text)) >= 2
        )
    )

    # Whole-sentence box + one-column fragment: the shorter OCR can have one or
    # two wrong glyphs, so ordinary similarity is often too low.  Geometry plus
    # coverage of the shorter text is the reliable signal.
    if nested_geometry and coverage >= cfg.nested_fragment_text_coverage:
        return True

    if strong_geometry:
        fuzzy_threshold = max(0.62, cfg.fuzzy_text_similarity_thresh - 0.12)
        return exact_or_substring or coverage >= 0.78 or similarity >= fuzzy_threshold

    # Non-overlapping boxes must be very close, aligned and textually near-identical.
    if a.vertical != b.vertical:
        return False
    near = _bbox_gap_ratio(a, b) <= cfg.duplicate_near_gap_ratio
    aligned = _axis_overlap_ratio(a, b) >= 0.58
    centers_close = center_ratio <= min(cfg.group_center_dist_ratio, 0.85)
    size_similar = area_ratio >= 0.55
    return near and aligned and centers_close and (
        size_similar
        and _bbox_gap_ratio(a, b) <= min(cfg.duplicate_near_gap_ratio, 0.08)
        and (exact_or_substring or similarity >= max(0.90, cfg.fuzzy_text_similarity_thresh))
    )


def _merge_group_masks(
    a: TextGroup,
    b: TextGroup,
    merged_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    if a.mask is None and b.mask is None:
        return None
    _x, _y, w, h = merged_bbox
    merged = np.zeros((h, w), dtype=np.uint8)
    _paste_group_mask(merged, a, merged_bbox)
    _paste_group_mask(merged, b, merged_bbox)
    return merged


def _best_ocr_group(a: TextGroup, b: TextGroup) -> TextGroup:
    a_norm = a.ocr_text_norm
    b_norm = b.ocr_text_norm
    if a_norm and b_norm:
        if a_norm in b_norm and len(b_norm) > len(a_norm):
            return b
        if b_norm in a_norm and len(a_norm) > len(b_norm):
            return a

        # In a nested whole-box/column pair, favor the more complete OCR unless
        # its quality is clearly worse.  Picking a tiny fragment just because its
        # confidence is 0.02 higher is what used to create incomplete/repeated text.
        if _text_coverage(a_norm, b_norm) >= 0.62 and abs(len(a_norm) - len(b_norm)) >= 2:
            longer, shorter = (a, b) if len(a_norm) > len(b_norm) else (b, a)
            if longer.ocr_confidence >= shorter.ocr_confidence - 0.16:
                return longer

    return max(
        (a, b),
        key=lambda group: (
            float(group.ocr_confidence) + min(0.12, len(group.ocr_text_norm) * 0.006),
            len(group.ocr_text_norm),
            _group_area(group),
        ),
    )


def _merge_group_objects(a: TextGroup, b: TextGroup) -> TextGroup:
    best = _best_ocr_group(a, b)
    merged_bbox = merge_bbox([a, b])
    translation_best = max(
        (a, b),
        key=lambda group: (
            bool(group.translation_valid and group.translation.strip()),
            len(group.translation),
            group.ocr_confidence,
        ),
    )
    merged_candidates: list[dict[str, object]] = []
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in a.ocr_candidates + b.ocr_candidates:
        key = (str(candidate.get("source", "")), str(candidate.get("normalized", "")))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        merged_candidates.append(candidate)

    return TextGroup(
        id=a.id,
        region_ids=list(dict.fromkeys(a.region_ids + b.region_ids)),
        bbox=merged_bbox,
        vertical=a.vertical if _group_area(a) >= _group_area(b) else b.vertical,
        ocr_text=best.ocr_text,
        ocr_text_norm=best.ocr_text_norm,
        ocr_confidence=best.ocr_confidence,
        ocr_source=best.ocr_source,
        ocr_candidates=merged_candidates,
        translation=translation_best.translation,
        translation_valid=translation_best.translation_valid,
        status=translation_best.status if translation_best.translation_valid else best.status,
        skip_reason="" if translation_best.translation_valid else best.skip_reason,
        sort_key=min(a.sort_key, b.sort_key),
        mapping_region_key=translation_best.mapping_region_key,
        mapping_chain=dict(translation_best.mapping_chain),
        mask=_merge_group_masks(a, b, merged_bbox),
    )


def _merge_duplicate_groups(
    groups: list[TextGroup],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_ocr_dedup or len(groups) < 2:
        return groups

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a_index: int, b_index: int) -> None:
        root_a, root_b = find(a_index), find(b_index)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _are_duplicate_groups(groups[i], groups[j], cfg):
                union(i, j)

    components: dict[int, list[TextGroup]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)

    merged: list[TextGroup] = []
    for component in components.values():
        current = component[0]
        for duplicate in component[1:]:
            duplicate.duplicate_of = current.id
            current = _merge_group_objects(current, duplicate)
        merged.append(current)
    return merged


def _are_translation_duplicates(
    a: TextGroup,
    b: TextGroup,
    cfg: PostprocessConfig,
) -> bool:
    """Second guard for OCR variants that translate to the same sentence."""
    if not (a.translation_valid and b.translation_valid):
        return False
    left = sanitize_translation_text(a.translation, source=a.ocr_text)
    right = sanitize_translation_text(b.translation, source=b.ocr_text)
    if not left or not right:
        return False

    translation_similarity = _text_similarity(left, right)
    if translation_similarity < 0.94:
        return False

    area_ratio = _group_area_ratio(a, b)
    bbox_containment = max(containment_ratio(a, b), containment_ratio(b, a))
    mask_containment = max(
        _group_mask_containment(a, b),
        _group_mask_containment(b, a),
    )
    strong_overlap = (
        _group_mask_iou(a, b) >= 0.42
        or mask_containment >= cfg.render_collision_mask_containment
        or (iom(a, b) >= max(0.65, cfg.same_text_iom_thresh) and area_ratio >= 0.45)
        or (bbox_containment >= cfg.containment_ratio_thresh and area_ratio >= 0.60)
    )
    if not strong_overlap:
        return False

    # 「嗯」「啊」等短句很常在相鄰氣泡合法重複，還要有來源文字相似訊號。
    compact = normalize_ocr_text(left, weak=True)
    return not (len(compact) <= 2 and _text_similarity(a.ocr_text_norm, b.ocr_text_norm) < 0.78)


def _merge_translation_duplicates(
    groups: list[TextGroup],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_ocr_dedup or len(groups) < 2:
        return groups

    kept: list[TextGroup] = []
    consumed = [False] * len(groups)
    for index, group in enumerate(groups):
        if consumed[index]:
            continue
        current = group
        for other_index in range(index + 1, len(groups)):
            if consumed[other_index]:
                continue
            other = groups[other_index]
            if not _are_translation_duplicates(current, other, cfg):
                continue
            other.duplicate_of = current.id
            current = _merge_group_objects(current, other)
            consumed[other_index] = True
        kept.append(current)
    return kept


def _group_source_rank(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> tuple[int, int, int]:
    sources = [
        regions_by_id[rid].source
        for rid in group.region_ids
        if rid in regions_by_id
    ]
    return (
        sum(source == "ctd" for source in sources),
        sum(source == "ctd_multiscale" for source in sources),
        -sum(source == "mask_fallback" for source in sources),
    )


def _render_group_score(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> tuple[object, ...]:
    local_mask = _local_group_mask(group)
    mask_pixels = int(np.count_nonzero(local_mask)) if local_mask is not None else 0
    return (
        _group_source_rank(group, regions_by_id),
        float(group.ocr_confidence),
        len(group.ocr_text_norm),
        mask_pixels,
        -_group_area(group),
    )


def _render_groups_conflict(
    a: TextGroup,
    b: TextGroup,
    cfg: PostprocessConfig,
) -> bool:
    """Prevent two translations from being drawn over the same text pixels.

    This guard intentionally does not trust OCR strings.  A whole-sentence mask
    and one nested column have low IoU and a tiny area ratio, yet nearly all pixels
    of the column are contained in the whole mask; mask containment catches it.
    """
    area_ratio = _group_area_ratio(a, b)
    mask_overlap = _group_mask_iou(a, b)
    mask_containment = max(
        _group_mask_containment(a, b),
        _group_mask_containment(b, a),
    )
    bbox_iom = iom(a, b)
    contained = max(containment_ratio(a, b), containment_ratio(b, a))
    nested_bbox = (
        a.vertical == b.vertical
        and contained >= cfg.nested_fragment_containment
        and area_ratio >= 0.08
    )
    return (
        mask_overlap >= cfg.render_collision_mask_iou
        or mask_containment >= cfg.render_collision_mask_containment
        or (bbox_iom >= cfg.render_collision_iom and area_ratio >= 0.50)
        or (contained >= cfg.render_collision_containment and area_ratio >= 0.55)
        or nested_bbox
    )


def _resolve_render_collisions(
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_render_collision_filter:
        return groups

    renderable = [
        group
        for group in groups
        if group.translation_valid and bool(group.translation.strip())
    ]
    ordered = sorted(
        renderable,
        key=lambda group: _render_group_score(group, regions_by_id),
        reverse=True,
    )
    accepted: list[TextGroup] = []
    for group in ordered:
        conflict = next(
            (
                winner
                for winner in accepted
                if _render_groups_conflict(group, winner, cfg)
            ),
            None,
        )
        if conflict is None:
            accepted.append(group)
            continue

        group.translation_valid = False
        group.translation = ""
        group.status = "render_collision_rejected"
        group.skip_reason = f"overlaps:{conflict.id}"
        group.duplicate_of = conflict.id
    return groups


def _layout_blocks_conflict(a: TextLayoutPlan, b: TextLayoutPlan) -> bool:
    """Final safety guard based on the text that will actually be drawn."""

    ax, ay, aw, ah = layout_plan_block_bbox(a)
    bx, by, bw, bh = layout_plan_block_bbox(b)
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    if overlap_w <= 0 or overlap_h <= 0:
        return False

    glyph_guard = max(2.0, min(a.font_size, b.font_size) * 0.24)
    if overlap_w < glyph_guard or overlap_h < glyph_guard:
        return False
    overlap_area = overlap_w * overlap_h
    return overlap_area / max(1, min(aw * ah, bw * bh)) >= 0.06


def _record_layout_plan(
    group: TextGroup,
    plan: TextLayoutPlan,
    *,
    mode: str,
) -> None:
    group.layout_bbox = plan.bbox
    group.rendered_font_size = int(plan.font_size)
    group.rendered_direction = plan.direction
    group.layout_mode = mode
    group.layout_info = plan.to_dict()
    group.layout_info["block_bbox"] = {
        key: value
        for key, value in zip(("x", "y", "w", "h"), layout_plan_block_bbox(plan))
    }


def _preflight_layout_plans(
    original: np.ndarray,
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    config: AppConfig,
) -> dict[str, TextLayoutPlan]:
    """Plan before erasing so a tiny/overlapping result keeps its original text.

    This is intentionally run before inpainting.  A layout that would need a font
    below the readability floor, or that would collide with a higher-quality
    subtitle, is rejected while the original Japanese pixels are still present.
    """

    renderable = [
        group
        for group in groups
        if group.translation_valid and bool(group.translation.strip())
    ]
    ordered = sorted(
        renderable,
        key=lambda group: _render_group_score(group, regions_by_id),
        reverse=True,
    )
    accepted: list[tuple[TextGroup, TextLayoutPlan]] = []
    plans: dict[str, TextLayoutPlan] = {}

    fallback_cfg = config.typesetting.model_copy(
        update={
            "adaptive_bubble_layout": False,
            "layout_padding_ratio": min(config.typesetting.layout_padding_ratio, 0.08),
        }
    )

    for group in ordered:
        plan = plan_text_layout(
            original,
            group,
            regions_by_id,
            group.translation,
            config.paths.font,
            config.typesetting,
            config.paths.font_fallback,
        )
        mode = config.typesetting.layout_mode
        if not plan.fits:
            group.translation_valid = False
            group.status = "layout_rejected"
            group.skip_reason = plan.reason or "unreadable_layout"
            _record_layout_plan(group, plan, mode=mode)
            continue

        collision = next(
            (winner for winner, winner_plan in accepted if _layout_blocks_conflict(plan, winner_plan)),
            None,
        )
        if collision is not None:
            compact = plan_text_layout(
                original,
                group,
                regions_by_id,
                group.translation,
                config.paths.font,
                fallback_cfg,
                config.paths.font_fallback,
            )
            compact_collision = (
                next(
                    (
                        winner
                        for winner, winner_plan in accepted
                        if compact.fits and _layout_blocks_conflict(compact, winner_plan)
                    ),
                    None,
                )
                if compact.fits
                else collision
            )
            if compact.fits and compact_collision is None:
                plan = compact
                mode = f"{config.typesetting.layout_mode}:no_bubble_expand"
            else:
                group.translation_valid = False
                group.status = "layout_collision_rejected"
                group.skip_reason = f"planned_text_overlaps:{collision.id}"
                _record_layout_plan(group, plan, mode=mode)
                continue

        _record_layout_plan(group, plan, mode=mode)
        accepted.append((group, plan))
        plans[group.id] = plan

    return plans


def _record_mapping_layout_plans(groups: list[TextGroup]) -> None:
    for group in groups:
        if group.layout_info and group.mapping_chain:
            group.mapping_chain["layout_plan"] = f"layout:{group.id}"


def _refresh_group_order(
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if cfg.reading_order == "jp_vertical":
        use_vertical_order = True
        groups = sorted(
            groups,
            key=lambda group: (
                -(group.x + group.w / 2.0),
                group.y + group.h / 2.0,
            ),
        )
    else:
        vertical_ratio = sum(group.vertical for group in groups) / max(1, len(groups))
        use_vertical_order = vertical_ratio >= 0.5
        if use_vertical_order:
            groups = sorted(
                groups,
                key=lambda group: (-(group.x + group.w / 2.0), group.y + group.h / 2.0),
            )
        else:
            groups = sorted(
                groups,
                key=lambda group: (group.y + group.h / 2.0, group.x + group.w / 2.0),
            )

    for index, group in enumerate(groups):
        group.id = f"g{index:03d}"
        if use_vertical_order:
            group.sort_key = (-(group.x + group.w / 2.0), group.y + group.h / 2.0)
        else:
            group.sort_key = (group.y + group.h / 2.0, group.x + group.w / 2.0)
        for region_id in group.region_ids:
            region = regions_by_id.get(region_id)
            if region is not None:
                region.group_id = group.id
    return groups


def _build_page_translation_units(groups: list[TextGroup]) -> tuple[list[TextGroup], list[str]]:
    # 呼叫者已經完成閱讀順序排序。此處不可再自行重排，否則回傳翻譯與
    # 原 group 逐項 zip 時可能錯置到別的對話框。
    ordered = list(groups)
    return ordered, [group.ocr_text for group in ordered]


def _mapping_region_key(page_id: str, group: TextGroup) -> str:
    identity_parts = sorted(group.region_ids) or [group.id]
    material = "|".join(
        [page_id, *identity_parts, *(str(value) for value in group.bbox)]
    )
    return "group:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _build_translation_request(
    groups: list[TextGroup],
    page_id: str,
) -> tuple[list[TextGroup], list[str], RequestMap]:
    ordered, texts = _build_page_translation_units(groups)
    for group in ordered:
        if not group.mapping_region_key:
            group.mapping_region_key = _mapping_region_key(page_id, group)
    request = build_request_map(
        page_id, ((group.mapping_region_key, group.ocr_text) for group in ordered)
    )
    return ordered, texts, request


def _request_translations(
    groups: list[TextGroup],
    page_id: str,
    config: AppConfig,
    glossary: dict[str, str],
) -> ValidatedTranslationBatch:
    _ordered, texts, request = _build_translation_request(groups, page_id)
    if not texts:
        return bind_validated_values(request, [])
    item_ids = [item.item_id for item in request.items]

    if not config.postprocess.enable_group_translate:
        if config.openrouter.translation_mode == "context":
            responses = translate_with_context_mapped(
                texts,
                config.openrouter,
                glossary,
                context_size=config.openrouter.context_size,
                item_ids=item_ids,
                artifact_root=config.paths.output_dir,
            )
        else:
            responses = translate_batch_mapped(
                texts,
                config.openrouter,
                glossary,
                item_ids=item_ids,
                artifact_root=config.paths.output_dir,
            )
        return bind_validated_responses(request, responses)

    total_chars = sum(len(text) for text in texts)
    should_fallback_window = total_chars > 6000 or len(texts) > 120
    if config.openrouter.page_context_mode == "page" and not should_fallback_window:
        responses = translate_page_mapped(
            texts,
            config.openrouter,
            glossary,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    elif config.openrouter.translation_mode == "context":
        responses = translate_with_context_mapped(
            texts,
            config.openrouter,
            glossary,
            context_size=config.openrouter.context_size,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    else:
        responses = translate_batch_mapped(
            texts,
            config.openrouter,
            glossary,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    return bind_validated_responses(request, responses)


def _translate_groups(
    groups: list[TextGroup],
    page_id: str,
    config: AppConfig,
    glossary: dict[str, str],
) -> ResultIssue | None:
    translatable: list[TextGroup] = []
    for group in groups:
        if group.status in {"ocr_rejected", "ocr_failed"}:
            continue
        accepted = bool(group.ocr_text_norm) and (
            group.ocr_confidence >= config.ocr.min_quality_score
            or not config.ocr.reject_non_japanese_noise
        )
        if not accepted:
            group.status = "ocr_rejected"
            group.skip_reason = "empty_ocr"
            if group.ocr_text_norm:
                group.skip_reason = f"low_ocr_quality:{group.ocr_confidence:.3f}"
            continue
        group.status = "ocr_accepted"
        translatable.append(group)

    if not translatable:
        return None

    _ordered, _texts, request = _build_translation_request(translatable, page_id)
    for group in translatable:
        request_item = request.by_region_key[group.mapping_region_key]
        group.mapping_chain = mapping_chain_template(
            region_key=group.mapping_region_key,
            ocr_record=f"ocr:{group.mapping_region_key}",
            request_item=request_item.item_id,
        )

    try:
        translations = _request_translations(translatable, page_id, config, glossary)
    except Exception as error:  # noqa: BLE001 - page boundary must preserve source on failure
        console.print(f"[red]本頁翻譯失敗，保留原文：{error}[/]")
        for group in translatable:
            group.translation = ""
            group.translation_valid = False
            group.status = "translation_failed"
            group.skip_reason = str(error)
        code = (
            "translation_mapping_failed"
            if isinstance(error, MappingContractError)
            else "translation_api_failed"
        )
        details: dict[str, object] = {}
        if isinstance(error, MappingContractError) and error.raw_response_refs:
            details["raw_response_artifacts"] = [
                reference.to_dict() for reference in error.raw_response_refs
            ]
        return ResultIssue(
            code=code,
            message=str(error),
            stage="translation",
            page_id=page_id,
            details=details,
        )

    for group in translatable:
        raw_translation = translations.by_region_key[group.mapping_region_key]
        translation = sanitize_translation_text(raw_translation, source=group.ocr_text)
        validation = validate_translation(group.ocr_text, translation, config.openrouter)
        group.mapping_chain = translations.chain_for(group.mapping_region_key)
        group.translation = translation if validation.valid else ""
        group.translation_valid = validation.valid
        if validation.valid:
            group.mapping_chain["validated_translation"] = hashlib.sha256(
                group.translation.encode("utf-8")
            ).hexdigest()
            group.status = "ready"
            group.skip_reason = ""
        else:
            group.mapping_chain["validated_translation"] = None
            group.status = "translation_rejected"
            group.skip_reason = ",".join(validation.issues) or "empty_translation"
        console.print(
            f"  [{group.id}] {group.ocr_text} → {group.translation or '[保留原文]'}",
            markup=False,
        )
    return None


def _mapping_snapshots(
    request_groups: list[TextGroup],
    final_groups: list[TextGroup],
) -> list[GroupMappingSnapshot]:
    """Preserve every request outcome while enriching surviving groups downstream."""

    def identity(group: TextGroup) -> tuple[str, str]:
        request_item = group.mapping_chain.get("request_item")
        if isinstance(request_item, str) and request_item:
            return ("request_item", request_item)
        if group.mapping_region_key:
            return ("region", group.mapping_region_key)
        return ("group", group.id)

    tracked = {identity(group): group for group in request_groups}
    for group in final_groups:
        tracked[identity(group)] = group
    return [GroupMappingSnapshot.from_group(group) for group in tracked.values()]


def _group_failure_issues(
    groups: list[TextGroup],
    page_id: str,
) -> list[ResultIssue]:
    failure_kinds = {
        "ocr_failed": ("ocr_group_failed", "ocr"),
        "translation_rejected": ("translation_rejected", "translation"),
        "layout_rejected": ("layout_rejected", "layout"),
        "layout_collision_rejected": ("layout_collision_rejected", "layout"),
    }
    issues: list[ResultIssue] = []
    for group in groups:
        failure = failure_kinds.get(group.status)
        if failure is None:
            continue
        code, stage = failure
        reason = group.skip_reason or group.status
        issues.append(
            ResultIssue(
                code=code,
                message=reason,
                stage=stage,
                page_id=page_id,
                details={
                    "group_id": group.id,
                    "group_status": group.status,
                    "reason": reason,
                    "region_ids": list(group.region_ids),
                },
            )
        )
    return issues


def _dump_debug_artifacts(
    image_path: Path,
    config: AppConfig,
    original_img: np.ndarray,
    detection: DetectionResult,
    groups: list[TextGroup],
    inpainted_img: np.ndarray | None,
    final_img: np.ndarray | None,
    dump_json: bool,
) -> None:
    dump_debug_artifacts(
        output_dir=config.paths.output_dir,
        page_name=image_path.name,
        original_image=original_img,
        regions_raw=detection.regions_raw,
        regions_post=detection.regions_post,
        groups=groups,
        save_json=dump_json,
        save_overlays=True,
        inpainted_image=inpainted_img,
        final_image=final_img,
    )


def _page_id_for_path(image_path: Path) -> str:
    try:
        with image_path.open("rb") as source_file:
            return hashlib.file_digest(source_file, "sha256").hexdigest()
    except OSError:
        return hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()


def process_single_page(
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    debug: bool = False,
    dump_json: bool = False,
    save_intermediate: bool = False,
    prep_manual: bool = False,
    *,
    page_id: str | None = None,
) -> PageResult:
    page_id = page_id or _page_id_for_path(image_path)
    with profile_page(page_id, str(image_path)):
        return _process_single_page_impl(
            image_path,
            config,
            glossary,
            page_id=page_id,
            debug=debug,
            dump_json=dump_json,
            save_intermediate=save_intermediate,
            prep_manual=prep_manual,
        )


def _process_single_page_impl(
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    *,
    page_id: str,
    debug: bool,
    dump_json: bool,
    save_intermediate: bool,
    prep_manual: bool,
) -> PageResult:
    with profile_span("decode"):
        image = read_image(image_path)
    if image is None:
        message = f"無法讀取圖片：{image_path}"
        return PageResult(
            page_id=page_id,
            source_path=image_path,
            status="failed",
            issues=[
                ResultIssue(
                    code="image_read_failed",
                    message=message,
                    stage="decode",
                    page_id=page_id,
                )
            ],
            stage_failure="decode",
        )
    original = image.copy()
    console.print(f"\n[bold]處理：{image_path.name}[/]")

    with profile_span("detection"):
        detection = detect_text_regions(image, config.detection, config.postprocess)
    set_page_profile_metrics(
        page_id,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        detected_groups=len(detection.groups),
    )
    detection_issues = [
        ResultIssue(
            code=issue.code,
            message=issue.message,
            stage="detection",
            page_id=page_id,
            details=issue.details,
        )
        for issue in detection.issues
    ]
    fallback_count = sum(region.source == "mask_fallback" for region in detection.regions_raw)
    console.print(
        f"  raw={len(detection.regions_raw)} post={len(detection.regions_post)} "
        f"groups={len(detection.groups)} mask-fallback={fallback_count}"
    )

    regions_by_id = {region.id: region for region in detection.regions_post}
    groups = list(detection.groups)
    if groups:
        try:
            initialize_ocr_model()
        except OCRInitializationError as error:
            return PageResult(
                page_id=page_id,
                source_path=image_path,
                status="blocked",
                source_image=original,
                regions=detection.regions_post,
                mapping_chains=[GroupMappingSnapshot.from_group(group) for group in groups],
                issues=[
                    *detection_issues,
                    ResultIssue(
                        code="ocr_initialization_failed",
                        message=str(error),
                        stage="ocr",
                        page_id=page_id,
                    )
                ],
                stage_failure="ocr",
            )

    for group in groups:
        try:
            with profile_span("ocr_group", group_id=group.id):
                ocr_result = ocr_group_detailed(
                    image=original,
                    group=group,
                    regions_by_id=regions_by_id,
                    cfg=config.ocr,
                    image_key=str(image_path.resolve()),
                )
            group.ocr_text = ocr_result.text
            group.ocr_confidence = ocr_result.confidence
            group.ocr_source = ocr_result.source
            group.ocr_candidates = [candidate.to_dict() for candidate in ocr_result.candidates]
            group_regions = [
                regions_by_id[rid]
                for rid in group.region_ids
                if rid in regions_by_id
            ]
            fallback_only = bool(group_regions) and all(
                region.source == "mask_fallback" for region in group_regions
            )
            has_pixel_mask = (
                group.mask is not None
                and group.mask.size > 0
                and bool(np.any(group.mask))
            )
            accepted, reason = assess_ocr_result(
                ocr_result,
                config.ocr,
                fallback_only=fallback_only,
            )
            if not has_pixel_mask:
                accepted, reason = False, "missing_text_mask"

            group.ocr_text_norm = (
                normalize_ocr_text(ocr_result.text, weak=True) if accepted else ""
            )
            group.status = "ocr_done" if accepted else "ocr_rejected"
            group.skip_reason = "" if accepted else reason
        except OCRInitializationError:
            raise
        except Exception as error:  # noqa: BLE001 - isolate OCR failure to this region
            group.ocr_text = ""
            group.ocr_text_norm = ""
            group.ocr_confidence = 0.0
            group.ocr_source = "error"
            group.status = "ocr_failed"
            group.skip_reason = str(error)
            console.print(f"[yellow]  [{group.id}] OCR 失敗，保留原文：{error}[/]")

    groups = _merge_duplicate_groups(groups, config.postprocess)
    groups = _refresh_group_order(groups, regions_by_id, config.postprocess)
    with profile_span("translation", group_count=len(groups)):
        translation_issue = _translate_groups(groups, page_id, config, glossary)
    mapping_outcomes = list(groups)
    groups_after_translation = _merge_translation_duplicates(groups, config.postprocess)
    if len(groups_after_translation) != len(groups):
        console.print(
            f"[yellow]  翻譯後再合併 {len(groups) - len(groups_after_translation)} 個強重疊重複框[/]"
        )
    groups_after_collision = _resolve_render_collisions(
        groups_after_translation,
        regions_by_id,
        config.postprocess,
    )
    collision_count = sum(
        group.status == "render_collision_rejected"
        for group in groups_after_collision
    )
    if collision_count:
        console.print(
            f"[yellow]  阻止 {collision_count} 個強重疊譯文寫入同一位置[/]"
        )
    groups = _refresh_group_order(
        groups_after_collision,
        regions_by_id,
        config.postprocess,
    )

    # 排版先在原圖上完整預演。放不下、會縮得過小或實際文字塊會互撞時，
    # 直接保留原文；不能先擦掉再發現無法安全寫回。
    with profile_span("layout", group_count=len(groups)):
        layout_plans = _preflight_layout_plans(
            original,
            groups,
            regions_by_id,
            config,
        )
    _record_mapping_layout_plans(groups)
    layout_rejected = sum(
        group.status in {"layout_rejected", "layout_collision_rejected"}
        for group in groups
    )
    if layout_rejected:
        console.print(
            f"[yellow]  {layout_rejected} 個譯文無法以接近原字級安全排版，已保留原文[/]"
        )

    # Inpainter 會依 translation_valid 過濾；OCR／翻譯／排版失敗的原文都不會被擦掉。
    detection.groups = groups
    with profile_span("inpaint"):
        inpainted = inpaint_regions(original, detection, config.inpainting)
    result = inpainted.copy()

    renderable = [
        group
        for group in groups
        if group.translation_valid and group.translation.strip() and group.id in layout_plans
    ]
    for group in renderable:
        with profile_span("render", group_id=group.id):
            result = render_text_into_group(
                image=result,
                group=group,
                regions_by_id=regions_by_id,
                text=group.translation,
                font_path=config.paths.font,
                cfg=config.typesetting,
                fallback_font_path=config.paths.font_fallback,
                layout_plan=layout_plans[group.id],
                layout_reference_image=original,
            )
        group.mapping_chain["render_target"] = f"render:{group.id}"

    unresolved = [group for group in groups if not group.translation_valid]
    if unresolved:
        console.print(
            f"[yellow]  {len(unresolved)} 個候選未通過 OCR／翻譯／排版檢查，已保留原文；"
            "可用 --debug --dump-json 查看原因。[/]"
        )

    if prep_manual or save_intermediate:
        intermediate_dir = config.paths.output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        write_image(intermediate_dir / f"{image_path.stem}_original.png", original)
        write_image(intermediate_dir / f"{image_path.stem}_inpainted.png", inpainted)
        write_image(intermediate_dir / f"{image_path.stem}_blanked.png", inpainted)

    if debug or dump_json or prep_manual:
        _dump_debug_artifacts(
            image_path=image_path,
            config=config,
            original_img=original,
            detection=detection,
            groups=groups,
            inpainted_img=inpainted if (debug or save_intermediate or prep_manual) else None,
            final_img=result if debug else None,
            dump_json=(dump_json or prep_manual),
        )

    group_issues = _group_failure_issues(groups, page_id)
    issues = [*detection_issues, *group_issues]
    if translation_issue is not None:
        issues.append(translation_issue)
    blocking_issue = translation_issue or (group_issues[0] if group_issues else None)
    set_page_profile_metrics(page_id, final_groups=len(groups), renderable_groups=len(renderable))
    return PageResult(
        page_id=page_id,
        source_path=image_path,
        status="blocked" if blocking_issue is not None else "succeeded",
        image=result,
        source_image=original,
        regions=detection.regions_post,
        ocr_results=[group.ocr_text for group in groups],
        translations=[group.translation for group in groups],
        mapping_chains=_mapping_snapshots(mapping_outcomes, groups),
        issues=issues,
        stage_failure=blocking_issue.stage if blocking_issue is not None else None,
    )


def run_pipeline(
    config: AppConfig,
    debug: bool = False,
    dump_json: bool = False,
    save_intermediate: bool = False,
    prep_manual: bool = False,
) -> BatchResult:
    output_dir = config.paths.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        issue = ResultIssue(code="output_write_failed", message=str(error), stage="output")
        return BatchResult(status="failed", pages=[], issues=[issue])

    image_files = get_image_files(config.paths.input_dir)
    if not image_files:
        message = f"在 {config.paths.input_dir} 找不到任何圖片檔"
        console.print(f"[red]{message}[/]")
        batch = BatchResult(
            status="failed",
            pages=[],
            issues=[ResultIssue(code="no_input_files", message=message, stage="input")],
        )
        _write_batch_manifest(batch, output_dir)
        return batch

    glossary = load_glossary(config.paths.glossary)
    console.print(f"[bold]找到 {len(image_files)} 張圖片[/]")
    console.print(f"[bold]模型：{config.openrouter.model}[/]")
    console.print(f"[bold]輸出：{output_dir}[/]")

    try:
        initialize_ocr_model()
    except OCRInitializationError as error:
        pages = [
            PageResult(
                page_id=_page_id_for_path(image_path),
                source_path=image_path,
                status="blocked",
                issues=[
                    ResultIssue(
                        code="ocr_initialization_failed",
                        message=str(error),
                        stage="ocr",
                        page_id=_page_id_for_path(image_path),
                    )
                ],
                stage_failure="ocr",
            )
            for image_path in image_files
        ]
        for page in pages:
            _preserve_failed_source(page, output_dir)
        batch = BatchResult(status=derive_batch_status(pages), pages=pages)
        _write_batch_manifest(batch, output_dir)
        console.print(f"[red]OCR 初始化失敗：{error}[/]")
        return batch

    pages: list[PageResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("翻譯中...", total=len(image_files))
        for image_path in image_files:
            page_id = _page_id_for_path(image_path)
            with profile_page(page_id, str(image_path)):
                try:
                    page = process_single_page(
                        image_path=image_path,
                        config=config,
                        glossary=glossary,
                        debug=debug,
                        dump_json=dump_json,
                        save_intermediate=save_intermediate,
                        prep_manual=prep_manual,
                        page_id=page_id,
                    )
                except OCRInitializationError as error:
                    page = _failed_page_result(
                        image_path,
                        code="ocr_initialization_failed",
                        stage="ocr",
                        error=error,
                        blocked=True,
                    )
                except Exception as error:  # noqa: BLE001 - page boundary records typed failure
                    page = _failed_page_result(
                        image_path,
                        code="page_processing_failed",
                        stage="pipeline",
                        error=error,
                    )
                _persist_page_result(page, output_dir)
            page.image = None
            page.source_image = None
            pages.append(page)
            progress.advance(task)

    batch = BatchResult(status=derive_batch_status(pages), pages=pages)
    _write_batch_manifest(batch, output_dir)
    total_regions = sum(len(page.regions) for page in pages)
    console.print(
        f"\n[bold]批次狀態：{batch.status}；共處理 {len(pages)} 頁，"
        f"{total_regions} 個文字區域，失敗／阻塞 {len(batch.failed_pages)} 頁[/]"
    )
    console.print(f"[bold]輸出目錄：{output_dir}[/]")
    return batch


def _failed_page_result(
    image_path: Path,
    *,
    code: str,
    stage: str,
    error: Exception,
    blocked: bool = False,
) -> PageResult:
    page_id = _page_id_for_path(image_path)
    return PageResult(
        page_id=page_id,
        source_path=image_path,
        status="blocked" if blocked else "failed",
        issues=[ResultIssue(code=code, message=str(error), stage=stage, page_id=page_id)],
        stage_failure=stage,
    )


def _failed_output_path(output_dir: Path, source_path: Path) -> Path:
    return output_dir / "failed" / f"{source_path.stem}.source-preserved{source_path.suffix}"


def _preserve_failed_source(page: PageResult, output_dir: Path) -> None:
    fallback_path = _failed_output_path(output_dir, page.source_path)
    try:
        (output_dir / page.source_path.name).unlink(missing_ok=True)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(page.source_path, fallback_path)
    except OSError as error:
        page.status = "failed"
        page.stage_failure = "output"
        page.issues.append(
            ResultIssue(
                code="output_write_failed",
                message=str(error),
                stage="output",
                page_id=page.page_id,
            )
        )
        page.output_path = None
        page.source_preserved = False
        return
    page.output_path = fallback_path
    page.source_preserved = True


def _persist_page_result(page: PageResult, output_dir: Path) -> None:
    if not page.succeeded:
        _preserve_failed_source(page, output_dir)
        return

    output_path = output_dir / page.source_path.name
    try:
        if page.image is None:
            raise ImageEncodeError(f"沒有可編碼的結果圖片：{page.source_path}")
        with profile_span("encode", output_name=output_path.name):
            write_image_or_raise(output_path, page.image)
    except (ImageEncodeError, ImageWriteError) as error:
        code = "image_encode_failed" if isinstance(error, ImageEncodeError) else "output_write_failed"
        page.status = "failed"
        page.stage_failure = "encode" if isinstance(error, ImageEncodeError) else "output"
        page.issues.append(
            ResultIssue(
                code=code,
                message=str(error),
                stage=page.stage_failure,
                page_id=page.page_id,
            )
        )
        _preserve_failed_source(page, output_dir)
        return
    page.output_path = output_path


def _write_batch_manifest(batch: BatchResult, output_dir: Path) -> None:
    manifest_path = output_dir / "batch-manifest.json"
    temporary_path = output_dir / ".batch-manifest.json.tmp"
    try:
        temporary_path.write_text(
            json.dumps(batch.to_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        batch.issues.append(
            ResultIssue(code="output_write_failed", message=str(error), stage="manifest")
        )
        batch.status = "failed"
        batch.manifest_path = None
        return
    batch.manifest_path = manifest_path
