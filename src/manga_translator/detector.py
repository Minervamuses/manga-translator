"""漫畫文字區域偵測、mask 候選回收與幾何後處理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from rich.console import Console

from .config import DetectionConfig, PostprocessConfig
from .geometry import (
    bbox_touch_or_near,
    center_distance,
    containment_ratio,
    iom,
    iou,
    merge_bbox,
)
from .reading_order import sort_groups_auto, sort_groups_jp_vertical, sort_regions_jp_vertical

console = Console()
RegionSource = Literal["ctd", "ctd_multiscale", "mask_fallback"]


@dataclass
class TextRegion:
    id: str
    x: int
    y: int
    w: int
    h: int
    vertical: bool = False
    confidence: float = 1.0
    source: RegionSource = "ctd"
    raw_index: int = -1
    detection_input_size: int = 0
    font_size_hint: float = -1.0
    mask_bbox: tuple[int, int, int, int] | None = None
    # local_mask 的大小固定等於 (h, w)，避免每個 region 保存整頁 mask。
    local_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    group_id: str | None = None
    candidate_duplicate: bool = False

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass
class TextGroup:
    id: str
    region_ids: list[str]
    bbox: tuple[int, int, int, int]
    vertical: bool
    ocr_text: str = ""
    ocr_text_norm: str = ""
    ocr_confidence: float = 0.0
    ocr_source: str = ""
    ocr_candidates: list[dict[str, Any]] = field(default_factory=list)
    translation: str = ""
    translation_valid: bool = False
    status: str = "detected"
    skip_reason: str = ""
    duplicate_of: str | None = None
    sort_key: tuple[float, float] = (0.0, 0.0)
    # 最終排版資訊只供 debug／驗證，不參與偵測與去重。
    layout_bbox: tuple[int, int, int, int] | None = None
    rendered_font_size: int = 0
    rendered_direction: str = ""
    layout_mode: str = ""
    layout_info: dict[str, Any] = field(default_factory=dict)
    # group mask 是與 bbox 對齊的局部 mask，避免每個群組複製一張整頁遮罩。
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]


@dataclass
class DetectionResult:
    regions_raw: list[TextRegion]
    regions_post: list[TextRegion]
    groups: list[TextGroup]
    mask: np.ndarray
    raw_mask: np.ndarray | None = None
    raw_blocks: list[Any] = field(default_factory=list)

    @property
    def regions(self) -> list[TextRegion]:
        """向後相容：舊程式把 regions 視為最終可用框。"""
        return self.regions_post


_detector: Any | None = None
_detector_key: tuple[str, str, bool, float, float, float] | None = None
_forced_cpu_due_cuda_error = False


def _is_cuda_runtime_error(err: Exception) -> bool:
    msg = str(err).lower()
    patterns = (
        "cuda error",
        "no kernel image is available",
        "invalid device function",
        "device-side assert",
        "cuda out of memory",
        "out of memory",
    )
    return any(p in msg for p in patterns)


def _resolve_device(device: str) -> str:
    if _forced_cpu_due_cuda_error and device.lower() == "cuda":
        return "cpu"
    requested = device.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]偵測器要求 cuda，但目前不可用，改用 cpu[/]")
        return "cpu"
    return requested


def _get_detector(cfg: DetectionConfig):
    """同一套權重只載入一次；不同 input size 直接調整推論尺寸。"""
    global _detector, _detector_key
    from .ctd.inference import TextDetector as CTDTextDetector

    model_path = Path(cfg.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到 comic-text-detector 模型檔：{model_path}\n"
            "請先執行 scripts/download_models.sh"
        )

    device = _resolve_device(cfg.device)
    half = cfg.half and device == "cuda"
    key = (
        str(model_path),
        device,
        half,
        cfg.nms_thresh,
        cfg.conf_thresh,
        cfg.mask_thresh,
    )
    if _detector is not None and _detector_key == key:
        return _detector

    _detector = CTDTextDetector(
        model_path=str(model_path),
        input_size=cfg.input_size,
        device=device,
        half=half,
        nms_thresh=cfg.nms_thresh,
        conf_thresh=cfg.conf_thresh,
        mask_thresh=cfg.mask_thresh,
    )
    _detector_key = key
    return _detector


def _set_detector_input_size(detector: Any, input_size: int) -> None:
    detector.input_size = (int(input_size), int(input_size))


def _normalize_mask(mask: np.ndarray | None, img_h: int, img_w: int) -> np.ndarray:
    if mask is None:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def _conservative_text_mask(
    refined_mask: np.ndarray,
    raw_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """限制 refined mask 只能落在 raw segmentation 附近。

    comic-text-detector 的 annotation refinement 會嘗試補齊字形，但在漫畫上也可能
    把眼睛、頭髮、酒杯反光或網點線條一起納入。raw mask 的召回較保守，因此拿它
    當空間支撐，再只膨脹少量像素補回描邊，可避免後續修補模糊整塊人物線稿。
    """
    _, support = cv2.threshold(
        raw_mask,
        int(cfg.raw_support_threshold),
        255,
        cv2.THRESH_BINARY,
    )
    radius = max(0, int(cfg.raw_support_dilate))
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        support = cv2.dilate(support, kernel, iterations=1)

    refined_binary = (refined_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(refined_binary, support)


def _blocks_to_regions(
    blocks: list[Any],
    img_h: int,
    img_w: int,
    *,
    source: RegionSource,
    input_size: int,
    raw_index_offset: int = 0,
) -> list[TextRegion]:
    regions: list[TextRegion] = []
    for pass_idx, blk in enumerate(blocks):
        xyxy = getattr(blk, "xyxy", None)
        if xyxy is None or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue

        regions.append(
            TextRegion(
                id="",
                x=x1,
                y=y1,
                w=w,
                h=h,
                vertical=bool(getattr(blk, "vertical", False)),
                confidence=float(getattr(blk, "prob", 1.0) or 1.0),
                source=source,
                raw_index=raw_index_offset + pass_idx,
                detection_input_size=input_size,
                font_size_hint=float(getattr(blk, "font_size", -1.0) or -1.0),
                mask_bbox=(x1, y1, w, h),
            )
        )
    return regions


def _intersection_area(a: TextRegion, b: TextRegion) -> int:
    from .geometry import intersection_area

    return intersection_area(a, b)


def _iou(a: TextRegion, b: TextRegion) -> float:
    return iou(a, b)


def _iom(a: TextRegion, b: TextRegion) -> float:
    return iom(a, b)


def _containment_ratio(inner: TextRegion, outer: TextRegion) -> float:
    return containment_ratio(inner, outer)


def _mask_containment_ratio(inner: TextRegion, outer: TextRegion) -> float:
    """Return the fraction of ``inner`` text pixels also supported by ``outer``.

    Multi-scale CTD often returns one wide block for a complete vertical
    dialogue and one narrow block for every original column.  A rectangle-only
    containment test cannot distinguish that valid hierarchy from a loose,
    page-sized false-positive box.  Shared real text-mask pixels can.
    """

    if (
        inner.local_mask is None
        or outer.local_mask is None
        or inner.local_mask.size == 0
        or outer.local_mask.size == 0
    ):
        return 0.0

    inner_mask = inner.local_mask
    outer_mask = outer.local_mask
    if inner_mask.shape[:2] != (inner.h, inner.w):
        inner_mask = cv2.resize(
            inner_mask,
            (inner.w, inner.h),
            interpolation=cv2.INTER_NEAREST,
        )
    if outer_mask.shape[:2] != (outer.h, outer.w):
        outer_mask = cv2.resize(
            outer_mask,
            (outer.w, outer.h),
            interpolation=cv2.INTER_NEAREST,
        )

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

    inner_pixels = int(np.count_nonzero(inner_mask))
    if inner_pixels <= 0:
        return 0.0
    overlap_pixels = int(np.count_nonzero((inner_crop > 0) & (outer_crop > 0)))
    return overlap_pixels / inner_pixels


def _merge_bbox(regions: list[TextRegion]) -> tuple[int, int, int, int]:
    return merge_bbox(regions)


def _estimate_char_size(regions: list[TextRegion], img_h: int, img_w: int) -> int:
    hints = [r.font_size_hint for r in regions if 4 <= r.font_size_hint <= 256]
    if hints:
        return max(6, int(round(float(np.median(hints)))))

    # 沒有可靠提示時，以頁面短邊的約 1.8% 作為漫畫字級估計。
    return max(8, int(round(min(img_h, img_w) * 0.018)))


def _component_boxes(binary: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    boxes: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area_px = [int(v) for v in stats[label]]
        if area_px >= min_area and w > 1 and h > 1:
            boxes.append((x, y, w, h, area_px))
    return boxes


def _candidate_from_box(
    box: tuple[int, int, int, int, int],
    *,
    residual_mask: np.ndarray,
    cfg: DetectionConfig,
    img_h: int,
    img_w: int,
    vertical_hint: bool | None = None,
) -> TextRegion | None:
    x, y, w, h, _closed_area = box
    pad = cfg.mask_fallback_padding
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    w2, h2 = x2 - x1, y2 - y1
    if w2 <= 2 or h2 <= 2:
        return None

    bbox_area = w2 * h2
    if bbox_area > img_h * img_w * cfg.mask_fallback_max_area_ratio:
        return None

    pixels = int(np.count_nonzero(residual_mask[y1:y2, x1:x2]))
    if pixels < cfg.mask_fallback_min_area:
        return None

    # 過於稀疏通常是畫面雜訊，而不是完整文字區塊。
    density = pixels / max(1, bbox_area)
    if density < cfg.mask_fallback_min_density:
        return None

    local_mask = residual_mask[y1:y2, x1:x2].copy()
    component_count, _labels = cv2.connectedComponents((local_mask > 0).astype(np.uint8))
    if max(0, component_count - 1) < cfg.mask_fallback_min_components:
        return None

    return TextRegion(
        id="",
        x=x1,
        y=y1,
        w=w2,
        h=h2,
        vertical=vertical_hint if vertical_hint is not None else h2 > w2 * 1.10,
        confidence=0.25,
        source="mask_fallback",
        raw_index=-1,
        detection_input_size=0,
        font_size_hint=-1.0,
        mask_bbox=(x1, y1, w2, h2),
        local_mask=local_mask,
    )


def _dedupe_mask_candidates(candidates: list[TextRegion]) -> list[TextRegion]:
    """合併水平／垂直 morphology 產生的同一個 fallback 候選。"""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda r: r.area, reverse=True)
    kept: list[TextRegion] = []
    for candidate in ordered:
        duplicate_idx: int | None = None
        for idx, existing in enumerate(kept):
            area_ratio = min(candidate.area, existing.area) / max(candidate.area, existing.area, 1)
            if iou(candidate, existing) >= 0.58 or (
                area_ratio >= 0.55
                and (
                containment_ratio(candidate, existing) >= 0.88
                or containment_ratio(existing, candidate) >= 0.88
                )
            ):
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_idx]
        # 保留較緊的框；fallback OCR 對過大的空白範圍較敏感。
        if candidate.area < existing.area:
            kept[duplicate_idx] = candidate
    return kept


def _extract_mask_fallback_regions(
    raw_mask: np.ndarray,
    existing_regions: list[TextRegion],
    cfg: DetectionConfig,
) -> list[TextRegion]:
    """從 segmentation mask 回收 YOLO 未形成 block 的文字候選。"""
    if not cfg.mask_fallback_enabled:
        return []

    img_h, img_w = raw_mask.shape[:2]
    _, binary = cv2.threshold(
        raw_mask, cfg.mask_fallback_threshold, 255, cv2.THRESH_BINARY
    )

    # 已有 block 的範圍不再重建 fallback，避免同一句被 OCR/翻譯兩次。
    covered = np.zeros_like(binary)
    cover_pad = max(2, cfg.mask_fallback_padding // 2)
    for region in existing_regions:
        x1 = max(0, region.x - cover_pad)
        y1 = max(0, region.y - cover_pad)
        x2 = min(img_w, region.x + region.w + cover_pad)
        y2 = min(img_h, region.y + region.h + cover_pad)
        cv2.rectangle(covered, (x1, y1), (x2, y2), 255, -1)
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(covered))

    if np.count_nonzero(residual) < cfg.mask_fallback_min_area:
        return []

    char_size = _estimate_char_size(existing_regions, img_h, img_w)
    cross = max(3, int(round(char_size * 0.32)))
    # 日漫美術字、小假名與描邊字的字間距可能比一般內文字體大。
    along = max(7, int(round(char_size * 1.9)))

    kernels: list[tuple[np.ndarray, bool | None]] = [
        (cv2.getStructuringElement(cv2.MORPH_RECT, (along, cross)), False),
        (cv2.getStructuringElement(cv2.MORPH_RECT, (cross, along)), True),
        (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    max(3, int(round(char_size * 0.75))),
                    max(3, int(round(char_size * 0.75))),
                ),
            ),
            None,
        ),
    ]

    candidates: list[TextRegion] = []
    for kernel, vertical_hint in kernels:
        closed = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel, iterations=1)
        for box in _component_boxes(closed, cfg.mask_fallback_min_area):
            candidate = _candidate_from_box(
                box,
                residual_mask=residual,
                cfg=cfg,
                img_h=img_h,
                img_w=img_w,
                vertical_hint=vertical_hint,
            )
            if candidate is not None:
                candidates.append(candidate)

    return _dedupe_mask_candidates(candidates)


def _sort_regions_reading_order(
    regions: list[TextRegion],
    mode: Literal["jp_vertical", "auto"] = "jp_vertical",
) -> list[TextRegion]:
    if mode == "jp_vertical":
        return sort_regions_jp_vertical(regions)

    vertical_ratio = sum(1 for r in regions if r.vertical) / max(1, len(regions))
    if vertical_ratio >= 0.5:
        return sort_regions_jp_vertical(regions)
    return sorted(regions, key=lambda r: (r.y + r.h / 2.0, r.x + r.w / 2.0))


def _should_group(a: TextRegion, b: TextRegion, cfg: PostprocessConfig) -> bool:
    overlap = _iom(a, b) >= cfg.group_iom_thresh
    if overlap:
        small_area = min(a.area, b.area)
        large_area = max(a.area, b.area, 1)
        area_ratio = small_area / large_area

        # comic-text-detector 偶爾會產生一個涵蓋大半頁的低品質外框。
        # 若僅因包含關係就和每個小框 union，整頁台詞會在 OCR 前被吞成一群。
        # 真正的多尺寸重複框或同一句碎片通常不會有如此極端的面積差。
        if area_ratio < 0.08:
            return False

        contains = (
            containment_ratio(a, b) >= cfg.containment_ratio_thresh
            or containment_ratio(b, a) >= cfg.containment_ratio_thresh
        )
        if contains and area_ratio < 0.22:
            if a.area <= b.area:
                inner, outer = a, b
            else:
                inner, outer = b, a

            # A high-resolution pass often produces one spanning block around
            # all columns while the primary pass keeps one block per column.
            # Merge that hierarchy when the spanning block supports the same
            # actual text pixels.  This prevents OCR/translation from treating
            # each original column as a separate subtitle.
            same_text_pixels = _mask_containment_ratio(inner, outer) >= 0.72
            multiscale_pair = inner.detection_input_size != outer.detection_input_size
            aligned_span = (
                inner.vertical == outer.vertical
                and (
                    (
                        inner.vertical
                        and min(inner.h, outer.h) / max(inner.h, outer.h, 1) >= 0.45
                    )
                    or (
                        not inner.vertical
                        and min(inner.w, outer.w) / max(inner.w, outer.w, 1) >= 0.45
                    )
                )
            )
            if same_text_pixels and multiscale_pair and aligned_span:
                return True

            max_dim = max(a.w, a.h, b.w, b.h, 1)
            if center_distance(a, b) / max_dim > 0.35:
                return False

        # 方向不同而大小又明顯不一致時，多半是外框／畫面雜訊，不應先合併。
        if a.vertical != b.vertical and area_ratio < 0.55:
            return False
        return True

    if a.vertical != b.vertical:
        return False

    near = bbox_touch_or_near(a, b, pad=8)
    if not near:
        return False

    max_dim = max(a.w, a.h, b.w, b.h, 1)
    close = center_distance(a, b) / max_dim <= cfg.group_center_dist_ratio
    if not close:
        return False

    # 防止相鄰但屬於不同氣泡的框被 transitive union 串成一大群。
    if a.vertical:
        cross_overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        cross_ratio = cross_overlap / max(1, min(a.w, b.w))
    else:
        cross_overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        cross_ratio = cross_overlap / max(1, min(a.h, b.h))
    return cross_ratio >= 0.20


def _place_region_mask(
    target: np.ndarray,
    region: TextRegion,
    *,
    origin_x: int = 0,
    origin_y: int = 0,
) -> None:
    h, w = target.shape[:2]
    raw_x1 = region.x - origin_x
    raw_y1 = region.y - origin_y
    raw_x2 = raw_x1 + region.w
    raw_y2 = raw_y1 + region.h

    x1 = max(0, min(raw_x1, w))
    y1 = max(0, min(raw_y1, h))
    x2 = max(0, min(raw_x2, w))
    y2 = max(0, min(raw_y2, h))
    if x2 <= x1 or y2 <= y1:
        return

    if (
        region.local_mask is None
        or region.local_mask.size == 0
        or not np.any(region.local_mask)
    ):
        # 沒有可靠文字像素時寧可保留原文，也不能把整個偵測框（常包含臉或線稿）
        # 當成文字擦除。後續 pipeline 會把零 mask 群組標記為不可渲染。
        return

    local = region.local_mask
    if local.shape[:2] != (region.h, region.w):
        local = cv2.resize(local, (region.w, region.h), interpolation=cv2.INTER_NEAREST)

    source_x1 = max(0, x1 - raw_x1)
    source_y1 = max(0, y1 - raw_y1)
    source_x2 = source_x1 + (x2 - x1)
    source_y2 = source_y1 + (y2 - y1)
    crop = local[source_y1:source_y2, source_x1:source_x2]
    if crop.shape[:2] != (y2 - y1, x2 - x1):
        return
    target[y1:y2, x1:x2] = cv2.bitwise_or(target[y1:y2, x1:x2], crop)


def _build_local_group_mask(
    regions: list[TextRegion],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, w, h = bbox
    mask = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
    for region in regions:
        _place_region_mask(mask, region, origin_x=x, origin_y=y)
    return mask


def _build_groups(
    regions: list[TextRegion],
    image_shape: tuple[int, int],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not regions:
        return []

    del image_shape  # group mask 已改為 bbox-local，不再配置整頁遮罩。
    if not cfg.enable_grouping:
        groups: list[TextGroup] = []
        for i, region in enumerate(regions):
            group_id = f"g{i:03d}"
            region.group_id = group_id
            mask = _build_local_group_mask([region], region.bbox)
            groups.append(
                TextGroup(
                    id=group_id,
                    region_ids=[region.id],
                    bbox=region.bbox,
                    vertical=region.vertical,
                    mask=mask,
                )
            )
        return groups

    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a_idx: int, b_idx: int) -> None:
        root_a, root_b = find(a_idx), find(b_idx)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _should_group(regions[i], regions[j], cfg):
                union(i, j)

    components: dict[int, list[TextRegion]] = {}
    for i, region in enumerate(regions):
        components.setdefault(find(i), []).append(region)

    groups: list[TextGroup] = []
    for idx, component_regions in enumerate(components.values()):
        bbox = _merge_bbox(component_regions)
        vertical_votes = sum(1 for r in component_regions if r.vertical)
        vertical = vertical_votes >= (len(component_regions) / 2)
        group_id = f"g{idx:03d}"
        for region in component_regions:
            region.group_id = group_id

        mask = _build_local_group_mask(component_regions, bbox)

        groups.append(
            TextGroup(
                id=group_id,
                region_ids=[r.id for r in component_regions],
                bbox=bbox,
                vertical=vertical,
                mask=mask,
            )
        )

    return groups


def _mark_geometric_duplicates(regions: list[TextRegion], cfg: PostprocessConfig) -> None:
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            high_iom = _iom(a, b) >= cfg.same_text_iom_thresh
            contains = (
                _containment_ratio(a, b) >= cfg.containment_ratio_thresh
                or _containment_ratio(b, a) >= cfg.containment_ratio_thresh
            )
            if high_iom or contains:
                a.candidate_duplicate = True
                b.candidate_duplicate = True


def postprocess_regions(
    regions: list[TextRegion],
    image_shape: tuple[int, int],
    cfg: PostprocessConfig,
    refined_mask: np.ndarray | None = None,
) -> tuple[list[TextRegion], list[TextGroup]]:
    """候選框後處理：過濾、mask 綁定、候選重複標記與群組。"""
    img_h, img_w = image_shape
    filtered: list[TextRegion] = []
    for region in regions:
        if region.area < cfg.min_region_area:
            continue
        thin_ratio = min(region.w, region.h) / max(region.w, region.h)
        if thin_ratio < cfg.drop_thin_ratio:
            continue

        region.id = f"r{len(filtered):04d}"
        if region.local_mask is None and refined_mask is not None:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.w, region.y + region.h
            local = refined_mask[y1:y2, x1:x2].copy()
            if local.shape[:2] == (region.h, region.w) and np.any(local):
                region.local_mask = local
        filtered.append(region)

    _mark_geometric_duplicates(filtered, cfg)
    groups = _build_groups(filtered, image_shape, cfg)

    if cfg.reading_order == "jp_vertical":
        use_vertical_order = True
        groups = sort_groups_jp_vertical(groups)
    else:
        vertical_ratio = sum(1 for group in groups if group.vertical) / max(1, len(groups))
        use_vertical_order = vertical_ratio >= 0.5
        groups = sort_groups_auto(groups, vertical_ratio=vertical_ratio)

    region_map = {region.id: region for region in filtered}
    regions_post: list[TextRegion] = []
    for index, group in enumerate(groups):
        group.id = f"g{index:03d}"
        if use_vertical_order:
            group.sort_key = (-float(group.x + group.w / 2.0), float(group.y + group.h / 2.0))
        else:
            group.sort_key = (float(group.y + group.h / 2.0), float(group.x + group.w / 2.0))

        group_regions = [region_map[rid] for rid in group.region_ids if rid in region_map]
        ordered = _sort_regions_reading_order(group_regions, cfg.reading_order)
        for region in ordered:
            region.group_id = group.id
            regions_post.append(region)
        group.region_ids = [region.id for region in ordered]

    return regions_post, groups


def _run_detector_pass(
    detector: Any,
    image: np.ndarray,
    input_size: int,
    keep_undetected_mask: bool,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    from .ctd.utils.textmask import REFINEMASK_ANNOTATION

    _set_detector_input_size(detector, input_size)
    mask, mask_refined, blocks = detector(
        image,
        refine_mode=REFINEMASK_ANNOTATION,
        keep_undetected_mask=keep_undetected_mask,
    )
    return mask, mask_refined, list(blocks)


def detect_text_regions(
    image: np.ndarray,
    detection_cfg: DetectionConfig | None = None,
    postprocess_cfg: PostprocessConfig | None = None,
) -> DetectionResult:
    global _forced_cpu_due_cuda_error

    if detection_cfg is None:
        detection_cfg = DetectionConfig()
    if postprocess_cfg is None:
        postprocess_cfg = PostprocessConfig()

    img_h, img_w = image.shape[:2]
    detector = _get_detector(detection_cfg)

    pass_sizes = [detection_cfg.input_size]
    for size in detection_cfg.additional_input_sizes:
        if size not in pass_sizes:
            pass_sizes.append(size)

    all_regions: list[TextRegion] = []
    all_blocks: list[Any] = []
    raw_mask_union = np.zeros((img_h, img_w), dtype=np.uint8)
    refined_mask_union = np.zeros((img_h, img_w), dtype=np.uint8)
    primary_raw_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    raw_offset = 0

    for pass_index, input_size in enumerate(pass_sizes):
        try:
            mask, mask_refined, blocks = _run_detector_pass(
                detector,
                image,
                input_size,
                detection_cfg.keep_undetected_mask,
            )
        except RuntimeError as error:
            is_primary = pass_index == 0
            if is_primary and detection_cfg.device == "cuda" and _is_cuda_runtime_error(error):
                _forced_cpu_due_cuda_error = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                console.print("[yellow]偵測器 CUDA 失敗，後續自動改用 CPU：[/]")
                console.print(f"[yellow]{error}[/]")
                cpu_cfg = detection_cfg.model_copy(update={"device": "cpu", "half": False})
                detector = _get_detector(cpu_cfg)
                mask, mask_refined, blocks = _run_detector_pass(
                    detector,
                    image,
                    input_size,
                    detection_cfg.keep_undetected_mask,
                )
            elif not is_primary:
                if _is_cuda_runtime_error(error) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                console.print(
                    f"[yellow]略過額外偵測尺寸 {input_size}（推論失敗）：{error}[/]"
                )
                continue
            else:
                raise

        raw_mask = _normalize_mask(mask, img_h, img_w)
        refined_source = mask_refined if mask_refined is not None else mask
        refined_mask = _normalize_mask(refined_source, img_h, img_w)
        safe_mask = _conservative_text_mask(refined_mask, raw_mask, detection_cfg)
        if pass_index == 0:
            primary_raw_mask = raw_mask.copy()
        raw_mask_union = cv2.max(raw_mask_union, raw_mask)
        refined_mask_union = cv2.bitwise_or(refined_mask_union, safe_mask)

        source: RegionSource = "ctd" if pass_index == 0 else "ctd_multiscale"
        pass_regions = _blocks_to_regions(
            blocks,
            img_h,
            img_w,
            source=source,
            input_size=input_size,
            raw_index_offset=raw_offset,
        )
        raw_offset += len(blocks)
        all_regions.extend(pass_regions)
        all_blocks.extend(blocks)

    fallback_source_mask = (
        primary_raw_mask if detection_cfg.mask_fallback_primary_only else raw_mask_union
    )
    fallback_regions = _extract_mask_fallback_regions(
        fallback_source_mask,
        all_regions,
        detection_cfg,
    )
    all_regions.extend(fallback_regions)

    regions_post, groups = postprocess_regions(
        all_regions,
        (img_h, img_w),
        postprocess_cfg,
        refined_mask=refined_mask_union,
    )

    return DetectionResult(
        regions_raw=all_regions,
        regions_post=regions_post,
        groups=groups,
        mask=refined_mask_union,
        raw_mask=raw_mask_union,
        raw_blocks=all_blocks,
    )


def draw_debug_regions(
    image: np.ndarray,
    regions: list[TextRegion],
    groups: list[TextGroup] | None = None,
) -> np.ndarray:
    debug_img = image.copy()
    source_colors: dict[str, tuple[int, int, int]] = {
        "ctd": (0, 0, 255),
        "ctd_multiscale": (255, 0, 255),
        "mask_fallback": (0, 165, 255),
    }
    for index, region in enumerate(regions):
        color = source_colors.get(region.source, (255, 0, 0))
        cv2.rectangle(
            debug_img,
            (region.x, region.y),
            (region.x + region.w, region.y + region.h),
            color,
            2,
        )
        direction = "V" if region.vertical else "H"
        cv2.putText(
            debug_img,
            f"{index}:{direction}:{region.id}:{region.source}",
            (region.x, max(15, region.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )

    if groups:
        for group in groups:
            x, y, w, h = group.bbox
            color = (0, 255, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                debug_img,
                group.id,
                (x, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )
    return debug_img
