"""Debug artifacts 輸出（JSON + 偵測／狀態 overlay）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts.mapping import normalize_mapping_chain
from .domain.models import PageDocument
from .domain.serialization import canonical_document_bytes
from .image_io import write_image


def dump_page_document(output_dir: Path, page_name: str, document: PageDocument) -> Path:
    """Materialize the canonical manifest; debug JSON is never a second source of truth."""

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    destination = debug_dir / f"{Path(page_name).stem}_page_document.json"
    temporary = debug_dir / f".{destination.name}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_document_bytes(document))
            handle.flush()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _group_to_dict(group: Any) -> dict[str, Any]:
    """Compatibility projection for callers that inspect an in-memory legacy group."""

    x, y, width, height = group.bbox
    mask = getattr(group, "mask", None)
    return {
        "id": group.id,
        "region_ids": list(group.region_ids),
        "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
        "vertical": bool(getattr(group, "vertical", False)),
        "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
        "ocr_text": getattr(group, "ocr_text", ""),
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

    return outputs
