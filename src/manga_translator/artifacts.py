"""Debug artifacts 輸出（JSON + 偵測／狀態 overlay）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts.mapping import normalize_mapping_chain
from .image_io import write_image


def _region_to_dict(region: Any) -> dict[str, Any]:
    local_mask = getattr(region, "local_mask", None)
    return {
        "id": getattr(region, "id", ""),
        "x": int(region.x),
        "y": int(region.y),
        "w": int(region.w),
        "h": int(region.h),
        "vertical": bool(getattr(region, "vertical", False)),
        "confidence": float(getattr(region, "confidence", 1.0)),
        "source": getattr(region, "source", ""),
        "raw_index": int(getattr(region, "raw_index", -1)),
        "detection_input_size": int(getattr(region, "detection_input_size", 0)),
        "font_size_hint": float(getattr(region, "font_size_hint", -1.0)),
        "candidate_duplicate": bool(getattr(region, "candidate_duplicate", False)),
        "group_id": getattr(region, "group_id", None),
        "mask_pixels": int(np.count_nonzero(local_mask)) if local_mask is not None else 0,
    }


def _group_to_dict(group: Any) -> dict[str, Any]:
    x, y, w, h = group.bbox
    mask = getattr(group, "mask", None)
    return {
        "id": group.id,
        "region_ids": list(group.region_ids),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "vertical": bool(getattr(group, "vertical", False)),
        "sort_key": list(getattr(group, "sort_key", (0, 0))),
        "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
        "ocr_text": getattr(group, "ocr_text", ""),
        "ocr_text_norm": getattr(group, "ocr_text_norm", ""),
        "ocr_confidence": float(getattr(group, "ocr_confidence", 0.0)),
        "ocr_source": getattr(group, "ocr_source", ""),
        "ocr_candidates": getattr(group, "ocr_candidates", []),
        "translation": getattr(group, "translation", ""),
        "translation_valid": bool(getattr(group, "translation_valid", False)),
        "status": getattr(group, "status", ""),
        "skip_reason": getattr(group, "skip_reason", ""),
        "duplicate_of": getattr(group, "duplicate_of", None),
        "layout_bbox": (
            {
                "x": int(group.layout_bbox[0]),
                "y": int(group.layout_bbox[1]),
                "w": int(group.layout_bbox[2]),
                "h": int(group.layout_bbox[3]),
            }
            if getattr(group, "layout_bbox", None) is not None
            else None
        ),
        "rendered_font_size": int(getattr(group, "rendered_font_size", 0)),
        "rendered_direction": getattr(group, "rendered_direction", ""),
        "layout_mode": getattr(group, "layout_mode", ""),
        "layout_info": getattr(group, "layout_info", {}),
        "mapping_region_key": getattr(group, "mapping_region_key", ""),
        "mapping_chain": normalize_mapping_chain(getattr(group, "mapping_chain", {})),
    }


def draw_overlay_regions(image: np.ndarray, regions: list[Any], title: str = "") -> np.ndarray:
    overlay = image.copy()
    source_colors = {
        "ctd": (0, 0, 255),
        "ctd_multiscale": (255, 0, 255),
        "mask_fallback": (0, 165, 255),
    }
    for index, region in enumerate(regions):
        color = source_colors.get(getattr(region, "source", ""), (255, 0, 0))
        cv2.rectangle(
            overlay,
            (region.x, region.y),
            (region.x + region.w, region.y + region.h),
            color,
            2,
        )
        label = f"{index}:{getattr(region, 'id', '')}:{getattr(region, 'source', '')}"
        cv2.putText(
            overlay,
            label,
            (region.x, max(12, region.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )
    if title:
        cv2.putText(
            overlay,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return overlay


def draw_overlay_groups(image: np.ndarray, groups: list[Any]) -> np.ndarray:
    overlay = image.copy()
    status_colors = {
        "ready": (0, 200, 0),
        "ocr_rejected": (0, 165, 255),
        "ocr_failed": (0, 0, 255),
        "translation_failed": (0, 0, 255),
        "translation_rejected": (0, 0, 255),
        "render_collision_rejected": (128, 0, 255),
        "layout_rejected": (255, 0, 128),
    }
    for index, group in enumerate(groups):
        x, y, w, h = group.bbox
        status = getattr(group, "status", "")
        color = status_colors.get(status, (0, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        confidence = float(getattr(group, "ocr_confidence", 0.0))
        label = f"{index}:{group.id}:{status}:{confidence:.2f}"
        cv2.putText(
            overlay,
            label,
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )
    return overlay


def dump_debug_artifacts(
    output_dir: Path,
    page_name: str,
    original_image: np.ndarray,
    regions_raw: list[Any],
    regions_post: list[Any],
    groups: list[Any],
    save_json: bool = True,
    save_overlays: bool = True,
    inpainted_image: np.ndarray | None = None,
    final_image: np.ndarray | None = None,
) -> dict[str, Path]:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page_name).stem
    outputs: dict[str, Path] = {}

    if save_overlays:
        raw_overlay = draw_overlay_regions(original_image, regions_raw, title="raw")
        post_overlay = draw_overlay_regions(original_image, regions_post, title="post")
        group_overlay = draw_overlay_groups(original_image, groups)
        outputs["raw_overlay"] = debug_dir / f"{stem}_regions_raw.png"
        outputs["post_overlay"] = debug_dir / f"{stem}_regions_post.png"
        outputs["group_overlay"] = debug_dir / f"{stem}_groups.png"
        write_image(outputs["raw_overlay"], raw_overlay)
        write_image(outputs["post_overlay"], post_overlay)
        write_image(outputs["group_overlay"], group_overlay)
        if inpainted_image is not None:
            outputs["inpainted"] = debug_dir / f"{stem}_inpainted.png"
            write_image(outputs["inpainted"], inpainted_image)
        if final_image is not None:
            outputs["final"] = debug_dir / f"{stem}_final.png"
            write_image(outputs["final"], final_image)

    if save_json:
        manifest = {
            "page": page_name,
            "regions_raw": [_region_to_dict(region) for region in regions_raw],
            "regions_post": [_region_to_dict(region) for region in regions_post],
            "groups": [_group_to_dict(group) for group in groups],
            "reading_order": [group.id for group in groups],
            "ocr_text": {group.id: getattr(group, "ocr_text", "") for group in groups},
            "ocr_text_norm": {
                group.id: getattr(group, "ocr_text_norm", "") for group in groups
            },
            "translation": {
                group.id: getattr(group, "translation", "") for group in groups
            },
            "unresolved": [
                {
                    "id": group.id,
                    "status": getattr(group, "status", ""),
                    "reason": getattr(group, "skip_reason", ""),
                }
                for group in groups
                if not bool(getattr(group, "translation_valid", False))
            ],
        }
        outputs["manifest"] = debug_dir / f"{stem}_manifest.json"
        with open(outputs["manifest"], "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

    return outputs
