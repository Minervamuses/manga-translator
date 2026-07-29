"""Generate the real five-page RAQM visual corpus and blind-review inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga_translator.benchmark.visual import (
    VisualMetrics,
    build_review_sheet,
    build_visual_report,
    write_group_bundle,
)
from manga_translator.config import DetectionConfig, InpaintingConfig, PostprocessConfig
from manga_translator.detector import detect_text_regions
from manga_translator.inpainter import inpaint_roi
from manga_translator.style.extract import extract_style_fingerprint
from manga_translator.typography.breaking import validate_breaks
from manga_translator.typography.fonts import FontResolver, FontRole
from manga_translator.typography.layout import (
    FontChoice,
    LayoutDirection,
    LayoutOverflow,
    LayoutRequest,
)
from manga_translator.typography.render import (
    AtomicRoiRequest,
    RenderStyle,
    atomic_inpaint_render,
    color_contrast_ratio,
    conservative_render_style,
    fit_render_style,
    render_layout_layer,
)
from manga_translator.typography.safe_region import build_safe_region
from manga_translator.typography.solver import (
    PillowLayoutRasterizer,
    estimate_source_line_gap_em,
    solve_layout,
    source_line_gap_options,
    text_block_bbox,
)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(path.suffix or ".png", image)
    if not encoded:
        raise ValueError(f"could not encode {path}")
    path.write_bytes(payload.tobytes())


def _fit_preview(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 232, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _materialize_blind_review_assets(
    root: Path,
    output: Path,
    sheet: dict[str, Any],
    key: dict[str, Any],
) -> list[dict[str, Any]]:
    key_rows = {row["blind_id"]: row for row in key["rows"]}
    contact_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for row in sheet["rows"]:
        blind_id = str(row["blind_id"])
        secret = key_rows[blind_id]
        previews = []
        for variant in ("a", "b"):
            source = root / str(secret[f"source_{variant}"])
            destination = root / str(row[f"preview_{variant}"])
            image = _read_image(source)
            _write_image(destination, image)
            previews.append(image)
        contact_rows.append((blind_id, previews[0], previews[1]))

    sheet_entries: list[dict[str, Any]] = []
    rows_per_sheet = 5
    panel_width, panel_height = 480, 300
    row_height = panel_height + 58
    for sheet_index in range(math.ceil(len(contact_rows) / rows_per_sheet)):
        subset = contact_rows[
            sheet_index * rows_per_sheet : (sheet_index + 1) * rows_per_sheet
        ]
        canvas = np.full(
            (52 + len(subset) * row_height, 24 + panel_width * 2 + 24, 3),
            248,
            dtype=np.uint8,
        )
        cv2.putText(
            canvas,
            f"Blind review {sheet_index + 1:02d}",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        ids = []
        for row_index, (blind_id, preview_a, preview_b) in enumerate(subset):
            ids.append(blind_id)
            top = 52 + row_index * row_height
            cv2.putText(
                canvas,
                f"{blind_id}   A",
                (24, top + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "B",
                (24 + panel_width + 24, top + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            canvas[
                top + 38 : top + 38 + panel_height,
                24 : 24 + panel_width,
            ] = _fit_preview(preview_a, panel_width, panel_height)
            canvas[
                top + 38 : top + 38 + panel_height,
                24 + panel_width + 24 : 24 + panel_width * 2 + 24,
            ] = _fit_preview(preview_b, panel_width, panel_height)
        contact_path = output / "review_sheets" / f"sheet_{sheet_index + 1:02d}.png"
        _write_image(contact_path, canvas)
        sheet_entries.append(
            {
                "sheet": _relative(root, contact_path),
                "blind_ids": ids,
            }
        )
    (output / "review_index.json").write_text(
        json.dumps(
            {"schema_version": "blind_review_index.v1", "sheets": sheet_entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return sheet_entries


def _materialize_review_assets(
    root: Path,
    output: Path,
    sheet: dict[str, Any],
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    source_rows = {row["review_id"]: row for row in sources["rows"]}
    contact_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for row in sheet["rows"]:
        review_id = str(row["review_id"])
        source = source_rows[review_id]
        legacy = _read_image(root / str(source["source_legacy"]))
        new = _read_image(root / str(source["source_new"]))
        _write_image(root / str(row["legacy_preview"]), legacy)
        _write_image(root / str(row["new_preview"]), new)
        contact_rows.append((review_id, legacy, new))

    entries: list[dict[str, Any]] = []
    rows_per_sheet = 5
    panel_width, panel_height = 480, 300
    row_height = panel_height + 58
    for sheet_index in range(math.ceil(len(contact_rows) / rows_per_sheet)):
        subset = contact_rows[
            sheet_index * rows_per_sheet : (sheet_index + 1) * rows_per_sheet
        ]
        canvas = np.full(
            (52 + len(subset) * row_height, 24 + panel_width * 2 + 24, 3),
            248,
            dtype=np.uint8,
        )
        cv2.putText(
            canvas,
            f"T1 review {sheet_index + 1:02d}",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        review_ids = []
        for row_index, (review_id, legacy, new) in enumerate(subset):
            review_ids.append(review_id)
            top = 52 + row_index * row_height
            cv2.putText(
                canvas,
                f"{review_id}   Legacy",
                (24, top + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "New",
                (24 + panel_width + 24, top + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            canvas[top + 38 : top + 38 + panel_height, 24 : 24 + panel_width] = (
                _fit_preview(legacy, panel_width, panel_height)
            )
            canvas[
                top + 38 : top + 38 + panel_height,
                24 + panel_width + 24 : 24 + panel_width * 2 + 24,
            ] = _fit_preview(new, panel_width, panel_height)
        sheet_path = output / "review_sheets" / f"sheet_{sheet_index + 1:02d}.png"
        _write_image(sheet_path, canvas)
        entries.append({"sheet": _relative(root, sheet_path), "review_ids": review_ids})
    (output / "review_index.json").write_text(
        json.dumps(
            {"schema_version": "visual_review_index.v2", "sheets": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return entries


def _box(payload: dict[str, Any] | list[int]) -> tuple[int, int, int, int]:
    if isinstance(payload, list):
        return tuple(int(value) for value in payload)  # type: ignore[return-value]
    return (
        int(payload["x"]),
        int(payload["y"]),
        int(payload.get("width", payload.get("w"))),
        int(payload.get("height", payload.get("h"))),
    )


def _union_roi(
    source_bbox: tuple[int, int, int, int],
    layout_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    sx, sy, sw, sh = source_bbox
    lx, ly, lw, lh = layout_bbox
    left = max(0, min(sx, lx) - padding)
    top = max(0, min(sy, ly) - padding)
    right = min(width, max(sx + sw, lx + lw) + padding)
    bottom = min(height, max(sy + sh, ly + lh) + padding)
    return (left, top, right - left, bottom - top)


def _derive_text_mask(crop: np.ndarray) -> np.ndarray:
    """Derive a sparse pixel mask from the immutable verified source crop."""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 2.2))
    nonzero = detail[detail > 0]
    threshold = max(10.0, float(np.percentile(nonzero, 72)) if nonzero.size else 10.0)
    edges = cv2.Canny(gray, 30, 90)
    mask = ((detail >= threshold) | (edges > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    maximum_component = max(12, round(mask.size * 0.35))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 2 <= area <= maximum_component:
            filtered[labels == label] = 255
    minimum_pixels = max(8, round(mask.size * 0.002))
    if np.count_nonzero(filtered) < minimum_pixels:
        filtered = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    if not np.any(filtered) or np.count_nonzero(filtered) >= filtered.size * 0.65:
        raise ValueError("verified source crop did not yield a safe non-rectangular text mask")
    return filtered


def _page_masks(
    root: Path,
    page_id: str,
    page_sha256: str,
    image: np.ndarray,
    regions: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    cache = root / "benchmarks" / "visual_v1" / "detector_masks"
    mask_path = cache / f"{page_id}.png"
    metadata_path = cache / f"{page_id}.json"
    model_path = (root / "models" / "comictextdetector.pt").resolve()
    detector_config = DetectionConfig(model_path=model_path)
    expected_metadata = {
        "schema_version": "visual_detector_mask.v1",
        "page_sha256": page_sha256,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_size": detector_config.input_size,
        "conf_thresh": detector_config.conf_thresh,
        "mask_thresh": detector_config.mask_thresh,
    }
    page_mask: np.ndarray | None = None
    if mask_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text("utf-8"))
        cached = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if metadata == expected_metadata and cached is not None and cached.shape == image.shape[:2]:
            page_mask = cached
    if page_mask is None:
        detection = detect_text_regions(
            image,
            detector_config,
            PostprocessConfig(),
        )
        page_mask = np.asarray(detection.mask, dtype=np.uint8)
        if page_mask.shape != image.shape[:2] or not np.any(page_mask):
            raise ValueError(f"real detector returned no aligned mask for page {page_id}")
        _write_image(mask_path, page_mask)
        metadata_path.write_text(
            json.dumps(expected_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    masks: dict[str, np.ndarray] = {}
    for region in regions:
        x, y, width, height = _box(region["bbox"])
        crop = image[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            raise ValueError(f"region {region['region_key']} lies outside its page")
        detector_crop = (page_mask[y : y + height, x : x + width] > 0).astype(np.uint8) * 255
        occupancy = np.count_nonzero(detector_crop) / detector_crop.size
        if not np.any(detector_crop) or occupancy >= 0.65:
            raise ValueError(
                f"real detector mask for {region['region_key']} is empty or non-sparse"
            )
        masks[str(region["region_key"])] = detector_crop
    return masks


def _paste_mask(
    target: np.ndarray,
    local: np.ndarray,
    bbox: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
) -> None:
    x, y, width, height = bbox
    left, top, roi_width, roi_height = roi
    x1, y1 = max(x, left), max(y, top)
    x2, y2 = min(x + width, left + roi_width), min(y + height, top + roi_height)
    if x2 <= x1 or y2 <= y1:
        return
    target[y1 - top : y2 - top, x1 - left : x2 - left] = np.maximum(
        target[y1 - top : y2 - top, x1 - left : x2 - left],
        local[y1 - y : y2 - y, x1 - x : x2 - x],
    )


def _render_style(style: Any, *, maximum_stroke_width: int) -> RenderStyle:
    return conservative_render_style(
        style,
        maximum_stroke_width=maximum_stroke_width,
    )


def _source_overlay(
    roi_image: np.ndarray,
    text_mask: np.ndarray,
    render_mask: np.ndarray,
) -> np.ndarray:
    overlay = roi_image.copy()
    text_contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    safe_contours, _ = cv2.findContours(
        render_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, safe_contours, -1, (0, 220, 0), 1)
    cv2.drawContours(overlay, text_contours, -1, (0, 220, 255), 1)
    return overlay


def _safe_visual(safe: Any, text_mask: np.ndarray) -> np.ndarray:
    visual = np.zeros((*safe.render_mask.shape, 3), dtype=np.uint8)
    visual[safe.safe_mask > 0] = (60, 100, 20)
    visual[safe.render_mask > 0] = (70, 210, 70)
    visual[safe.protected_edges > 0] = (30, 30, 230)
    visual[text_mask > 0] = (0, 220, 255)
    return visual


def _run_payload(run: Any) -> dict[str, Any]:
    payload = asdict(run)
    payload["glyph_coverage"] = [f"U+{value:04X}" for value in run.glyph_coverage]
    return payload


def _center(alpha: np.ndarray) -> tuple[float, float]:
    visible = np.argwhere(alpha > 0)
    if not visible.size:
        return (0.0, 0.0)
    return (float(np.mean(visible[:, 1])), float(np.mean(visible[:, 0])))


def _bbox_ratios(
    candidate_bbox: tuple[int, int, int, int],
    source_bbox: tuple[int, int, int, int],
    direction: LayoutDirection,
) -> tuple[float, float, float, float]:
    candidate_primary = candidate_bbox[3] if direction is LayoutDirection.VERTICAL else candidate_bbox[2]
    source_primary = source_bbox[3] if direction is LayoutDirection.VERTICAL else source_bbox[2]
    candidate_secondary = candidate_bbox[2] if direction is LayoutDirection.VERTICAL else candidate_bbox[3]
    source_secondary = source_bbox[2] if direction is LayoutDirection.VERTICAL else source_bbox[3]
    candidate_center = (
        candidate_bbox[0] + candidate_bbox[2] / 2.0,
        candidate_bbox[1] + candidate_bbox[3] / 2.0,
    )
    source_center = (
        source_bbox[0] + source_bbox[2] / 2.0,
        source_bbox[1] + source_bbox[3] / 2.0,
    )
    return (
        candidate_primary / max(1, source_primary),
        candidate_secondary / max(1, source_secondary),
        (candidate_bbox[2] * candidate_bbox[3]) / max(1, source_bbox[2] * source_bbox[3]),
        math.dist(candidate_center, source_center)
        / max(1.0, math.hypot(source_bbox[2], source_bbox[3])),
    )


def _reading_order_valid(layout: Any) -> bool:
    chunks = layout.candidate.chunks
    runs = layout.shaped_runs
    if len(chunks) <= 1:
        return True
    if len(runs) != len(chunks) or any(run.text != chunk for run, chunk in zip(runs, chunks)):
        return False
    if layout.candidate.direction is LayoutDirection.VERTICAL:
        return all(left.anchor[0] > right.anchor[0] for left, right in pairwise(runs))
    return all(upper.anchor[1] < lower.anchor[1] for upper, lower in pairwise(runs))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def generate(
    root: Path,
    font: Path,
    fallback: Path,
    *,
    selected_region_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    corpus = root / "benchmarks" / "regression_v032" / "pages"
    output = root / "benchmarks" / "visual_v1"
    resolver = FontResolver.from_paths(font, fallback)
    outline_width = 0
    maximum_extracted_stroke = 2
    rasterizer = PillowLayoutRasterizer(resolver, stroke_width=outline_width)
    inpainting = InpaintingConfig(method="hybrid")
    metrics: list[VisualMetrics] = []
    manifest_pages: list[dict[str, Any]] = []
    blind_groups: list[dict[str, Any]] = []

    for page_path in sorted(corpus.glob("*.json")):
        page = json.loads(page_path.read_text(encoding="utf-8"))
        page_id = str(page["page_id"])
        regions = list(page["regions"])
        if selected_region_keys is not None and not any(
            str(region["region_key"]) in selected_region_keys for region in regions
        ):
            continue
        source_path = root / str(page["source_image"])
        source = _read_image(source_path)
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != page["page_sha256"]:
            raise ValueError(f"page hash mismatch: {source_path}")
        legacy_rows = json.loads((root / str(page["fixture_source"])).read_text("utf-8"))
        legacy_by_id = {str(row["id"]): row for row in legacy_rows}
        legacy_preview_path = (
            root
            / str(page["fixture_source"]).replace("_layout.json", "_final_preview.jpg")
        )
        legacy_preview = _read_image(legacy_preview_path)
        local_masks = _page_masks(
            root,
            page_id,
            str(page["page_sha256"]),
            source,
            regions,
        )
        working = source.copy()
        occupied = np.zeros(source.shape[:2], dtype=np.uint8)
        group_entries: list[dict[str, Any]] = []

        for region in sorted(regions, key=lambda item: int(item["reading_order_index"])):
            region_key = str(region["region_key"])
            if selected_region_keys is not None and region_key not in selected_region_keys:
                continue
            legacy = legacy_by_id[str(region["legacy_group_id"])]
            bbox = _box(region["bbox"])
            layout_bbox = _box(legacy["layout"]["bbox"])
            source_font = max(1, int(legacy["layout"]["font_size"]))
            roi_bbox = _union_roi(
                bbox,
                layout_bbox,
                source.shape[:2],
                padding=max(10, round(source_font * 0.7)),
            )
            left, top, width, height = roi_bbox
            original_roi = source[top : top + height, left : left + width]
            text_mask = np.zeros((height, width), dtype=np.uint8)
            _paste_mask(text_mask, local_masks[region_key], bbox, roi_bbox)
            other_mask = np.zeros_like(text_mask)
            for other in regions:
                other_key = str(other["region_key"])
                if other_key != region_key:
                    _paste_mask(other_mask, local_masks[other_key], _box(other["bbox"]), roi_bbox)
            safe = build_safe_region(
                original_roi,
                text_mask,
                line_polygons=(
                    (
                        (bbox[0] - left, bbox[1] - top),
                        (bbox[0] + bbox[2] - left, bbox[1] - top),
                        (bbox[0] + bbox[2] - left, bbox[1] + bbox[3] - top),
                        (bbox[0] - left, bbox[1] + bbox[3] - top),
                    ),
                ),
                other_text_mask=other_mask,
            )
            if safe.confidence < 0.48 or not np.any(safe.render_mask):
                raise ValueError(
                    f"{region_key}: safe-region confidence {safe.confidence:.4f} is below 0.48"
                )
            translation = str(region["fixed_translation"])
            vertical = str(legacy["layout"]["direction"]) == "vertical"
            primary = LayoutDirection.VERTICAL if vertical else LayoutDirection.HORIZONTAL
            source_line_count = max(1, len(legacy["layout"].get("chunks", ())))
            source_block_global = _box(legacy["layout"].get("block_bbox", legacy["layout"]["bbox"]))
            source_block_bbox = (
                source_block_global[0] - left,
                source_block_global[1] - top,
                source_block_global[2],
                source_block_global[3],
            )
            source_center = (
                source_block_bbox[0] + source_block_bbox[2] / 2.0,
                source_block_bbox[1] + source_block_bbox[3] / 2.0,
            )
            legacy_line_step = float(legacy["layout"].get("primary_step", 0.0))
            source_line_gap_em = (
                legacy_line_step / source_font
                if source_line_count > 1 and legacy_line_step > 0
                else estimate_source_line_gap_em(
                    source_block_bbox,
                    primary,
                    source_font,
                    source_line_count,
                )
            )
            request = LayoutRequest(
                text=translation,
                safe_region=safe,
                fonts=(FontChoice(FontRole.NEUTRAL_SANS),),
                source_font_size=float(source_font),
                source_center=source_center,
                hard_font_floor=max(10, math.ceil(source_font * 0.90)),
                max_lines=source_line_count + 1,
                source_direction=primary,
                allow_alternate_direction=False,
                source_line_count=source_line_count,
                source_line_gap_em=source_line_gap_em,
                source_text_bbox=source_block_bbox,
                directions=(primary,),
                line_gap_options=source_line_gap_options(source_line_gap_em),
                neighbor_mask=occupied[top : top + height, left : left + width],
                minimum_containment=0.995,
            )
            layout = solve_layout(request, rasterizer)
            style = extract_style_fingerprint(
                source,
                text_mask,
                bbox=roi_bbox,
                source_angle=0.0,
            )
            before = working.copy()
            safety_rejection = layout.reason if isinstance(layout, LayoutOverflow) else None
            background = style.background
            background_rgb = (
                background.value
                if background is not None
                and background.status == "known"
                and background.value is not None
                and background.confidence >= 0.65
                else None
            )
            render_style = _render_style(style, maximum_stroke_width=maximum_extracted_stroke)
            contrast = (
                color_contrast_ratio(render_style.fill[:3], background_rgb)
                if background_rgb is not None
                else 0.0
            )
            if safety_rejection is None and background_rgb is None:
                safety_rejection = "unreliable_background_contrast"
            if safety_rejection is None and contrast < 4.5:
                safety_rejection = "insufficient_text_contrast"
            if safety_rejection is None and not _reading_order_valid(layout):
                safety_rejection = "invalid_reading_order"

            layer = np.zeros((height, width, 4), dtype=np.uint8)
            styled_containment = 1.0
            outside_changed = 0
            collision = 0
            candidate_bbox = source_block_bbox
            primary_ratio = secondary_ratio = area_ratio = 1.0
            center_ratio = 0.0
            reading_order_valid = True
            font_size = source_font
            candidate_direction = primary
            break_indices: tuple[int, ...] = ()
            shaped_runs: tuple[Any, ...] = ()
            plan_hash: str | None = None
            if safety_rejection is None:
                render_style = fit_render_style(layout, safe, render_style)
                layer = render_layout_layer(layout, render_style)
                styled_containment = safe.alpha_containment(layer[:, :, 3])
                if styled_containment < 0.995:
                    safety_rejection = "styled_alpha_outside_render_mask"
                else:
                    outcome = atomic_inpaint_render(
                        working,
                        AtomicRoiRequest(
                            roi_bbox,
                            text_mask,
                            safe,
                            layout,
                            render_style,
                            source_text_bbox=source_block_bbox,
                            background_rgb=background_rgb,
                        ),
                        inpainting,
                    )
                    if not outcome.committed:
                        safety_rejection = outcome.reason
            if safety_rejection is None:
                outside = np.ones(source.shape[:2], dtype=bool)
                outside[top : top + height, left : left + width] = False
                outside_changed = int(
                    np.count_nonzero(np.any(working[outside] != before[outside], axis=1))
                )
                occupied_roi = occupied[top : top + height, left : left + width]
                collision = int(np.any((layer[:, :, 3] > 0) & (occupied_roi > 0)))
                occupied_roi[layer[:, :, 3] > 0] = 255
                candidate_bbox = text_block_bbox(layout.alpha)
                primary_ratio, secondary_ratio, area_ratio, center_ratio = _bbox_ratios(
                    candidate_bbox,
                    source_block_bbox,
                    primary,
                )
                reading_order_valid = _reading_order_valid(layout)
                font_size = layout.candidate.font_size
                candidate_direction = layout.candidate.direction
                break_indices = layout.candidate.break_indices
                shaped_runs = layout.shaped_runs
                plan_hash = layout.plan_hash
            else:
                # The atomic path has not committed anything on failure; retaining
                # the source pixels is the successful safety outcome for this group.
                working[:] = before
                failure_dir = output / "failures" / region_key.replace(":", "-")
                _write_image(failure_dir / "source.png", original_roi)
                _write_image(failure_dir / "text_mask.png", text_mask)
                _write_image(failure_dir / "safe_mask.png", _safe_visual(safe, text_mask))

            accepted = safety_rejection is None
            inpainted = inpaint_roi(original_roi, text_mask, inpainting) if accepted else original_roi
            final_roi = working[top : top + height, left : left + width].copy()
            legacy_roi = legacy_preview[top : top + height, left : left + width]
            group_dir = output / "pages" / page_id / str(region["legacy_group_id"])
            safe_bbox = text_block_bbox(safe.render_mask)
            block_area = candidate_bbox[2] * candidate_bbox[3]
            safe_area = max(1, safe_bbox[2] * safe_bbox[3])
            font_floor_met = not accepted or font_size >= math.ceil(source_font * 0.90)
            block_bbox_valid = not accepted or (
                primary_ratio >= request.minimum_primary_axis_ratio
                and secondary_ratio >= request.minimum_secondary_axis_ratio
                and area_ratio >= request.minimum_text_block_area_ratio
                and center_ratio <= request.maximum_center_offset_ratio
            )
            bundle_metrics = VisualMetrics(
                page_id=page_id,
                region_key=region_key,
                mapping_complete=bool(region.get("verified_by")) and bool(translation),
                missing_glyphs=0,
                clreq_hard_violations=len(
                    validate_breaks(translation, break_indices)
                ),
                accepted_collisions=collision,
                outside_roi_changed_pixels=outside_changed,
                alpha_containment=styled_containment,
                font_size_ratio=font_size / source_font,
                center_offset_px=center_ratio
                * math.hypot(source_block_bbox[2], source_block_bbox[3]),
                whitespace_ratio=1.0 - block_area / safe_area,
                candidate_status="accepted" if accepted else "preserved_original",
                safety_rejection=safety_rejection,
                orientation_match=not accepted or candidate_direction is primary,
                reading_order_valid=reading_order_valid,
                font_floor_met=font_floor_met,
                text_block_bbox_valid=block_bbox_valid,
                contrast_valid=not accepted or contrast >= 4.5,
                no_erase_on_rejection=accepted or bool(np.array_equal(working, before)),
                primary_axis_ratio=primary_ratio,
                secondary_axis_ratio=secondary_ratio,
                text_block_area_ratio=area_ratio,
                center_offset_ratio=center_ratio,
                contrast_ratio=contrast if accepted else 21.0,
            )
            written = write_group_bundle(
                group_dir,
                source_overlay=_source_overlay(original_roi, text_mask, safe.render_mask),
                safe_mask=_safe_visual(safe, text_mask),
                style_fingerprint=style.model_dump(mode="json"),
                shaped_runs=(_run_payload(run) for run in shaped_runs),
                layout_alpha=layout.alpha if accepted else np.zeros_like(text_mask),
                inpainted_roi=inpainted,
                final_preview=final_roi,
                metrics=bundle_metrics,
            )
            legacy_output = group_dir / "legacy_preview.png"
            _write_image(legacy_output, legacy_roi)
            artifacts = {
                name: _relative(root, Path(path)) for name, path in written.items()
            }
            metrics.append(bundle_metrics)
            group_entries.append(
                {
                    "region_key": region_key,
                    "legacy_group_id": region["legacy_group_id"],
                    "roi_bbox": list(roi_bbox),
                    "safe_region_confidence": safe.confidence,
                    "layout_plan_hash": plan_hash,
                    "candidate_status": bundle_metrics.candidate_status,
                    "safety_rejection": safety_rejection,
                    "artifacts": artifacts,
                }
            )
            blind_groups.append(
                {
                    "page_id": page_id,
                    "region_key": region_key,
                    "legacy_preview": _relative(root, legacy_output),
                    "new_preview": artifacts["final_preview"],
                }
            )
            print(
                f"generated {region_key} status={bundle_metrics.candidate_status} "
                f"plan={(plan_hash or 'none')[:12]} confidence={safe.confidence:.4f}",
                flush=True,
            )

        if selected_region_keys is not None:
            continue
        page_preview = output / "pages" / page_id / "final_page.png"
        _write_image(page_preview, working)
        manifest_pages.append(
            {
                "page_id": page_id,
                "source_image": page["source_image"],
                "artifact_status": "generated",
                "detector_mask": _relative(
                    root, output / "detector_masks" / f"{page_id}.png"
                ),
                "groups": group_entries,
                "final_page": _relative(root, page_preview),
            }
        )

    if selected_region_keys is not None:
        return {
            "schema_version": "visual_v1.targeted_run",
            "group_count": len(metrics),
            "groups": [asdict(row) for row in metrics],
        }

    sheet, sources = build_review_sheet(blind_groups)
    _materialize_review_assets(root, output, sheet, sources)
    (output / "review_sheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_visual_report(metrics, reviews=sheet["rows"], required_pages=5)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "visual_v1.manifest",
        "status": "blocked",
        "automated_status": report["automated_status"],
        "page_count": len(manifest_pages),
        "group_count": len(metrics),
        "pages": manifest_pages,
        "blockers": [f"manual_review_0_of_{len(metrics)}"],
        "engine_switch": "waiting_for_t1_review",
        "report": "report.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_existing_blind_review_assets(root: Path) -> list[dict[str, Any]]:
    """Migrate an existing sheet to anonymous paths and rebuild contact sheets."""

    output = root / "benchmarks" / "visual_v1"
    sheet_path = output / "blind_review_sheet.json"
    key_path = output / "blind_review_key.json"
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    sheet_rows = {row["blind_id"]: row for row in sheet["rows"]}
    for secret in key["rows"]:
        row = sheet_rows[secret["blind_id"]]
        if "source_a" not in secret:
            secret["source_a"] = row["preview_a"]
            secret["source_b"] = row["preview_b"]
        blind_dir = f"benchmarks/visual_v1/blind_previews/{secret['blind_id']}"
        row["preview_a"] = f"{blind_dir}/a.png"
        row["preview_b"] = f"{blind_dir}/b.png"
    entries = _materialize_blind_review_assets(root, output, sheet, key)
    sheet_path.write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return entries


def refresh_existing_review_assets(root: Path) -> dict[str, Any]:
    """Refresh the v2 review/report index from existing per-group artifacts."""

    output = root / "benchmarks" / "visual_v1"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics: list[VisualMetrics] = []
    review_groups: list[dict[str, str]] = []
    for page in manifest["pages"]:
        final_page_path = root / str(page["final_page"])
        final_page = _read_image(final_page_path)
        final_page_changed = False
        for group in page["groups"]:
            metric_path = root / str(group["artifacts"]["metrics"])
            metric = VisualMetrics(**json.loads(metric_path.read_text(encoding="utf-8")))
            metrics.append(metric)
            group["candidate_status"] = metric.candidate_status
            if metric.safety_rejection is not None:
                group["safety_rejection"] = metric.safety_rejection
            if metric.candidate_status == "preserved_original":
                group["layout_plan_hash"] = None
                left, top, width, height = (int(value) for value in group["roi_bbox"])
                preserved = _read_image(root / str(group["artifacts"]["final_preview"]))
                final_page[top : top + height, left : left + width] = preserved
                final_page_changed = True
            review_groups.append(
                {
                    "page_id": str(page["page_id"]),
                    "region_key": str(group["region_key"]),
                    "legacy_preview": _relative(
                        root,
                        output
                        / "pages"
                        / str(page["page_id"])
                        / str(group["legacy_group_id"])
                        / "legacy_preview.png",
                    ),
                    "new_preview": str(group["artifacts"]["final_preview"]),
                }
            )
        if final_page_changed:
            _write_image(final_page_path, final_page)

    sheet, sources = build_review_sheet(review_groups)
    _materialize_review_assets(root, output, sheet, sources)
    (output / "review_sheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_visual_report(metrics, reviews=sheet["rows"], required_pages=5)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["automated_status"] = report["automated_status"]
    manifest["status"] = "blocked"
    manifest["blockers"] = [f"manual_review_0_of_{len(metrics)}"]
    manifest["engine_switch"] = "waiting_for_t1_review"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--font", type=Path, default=Path("fonts/Iansui-Regular.ttf"))
    parser.add_argument(
        "--fallback", type=Path, default=Path("fonts/NotoSansCJKtc-Regular.otf")
    )
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--region-key", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.review_only:
        print(
            json.dumps(
                refresh_existing_review_assets(root),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    manifest = generate(
        root,
        root / args.font,
        root / args.fallback,
        selected_region_keys=(frozenset(args.region_key) if args.region_key else None),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
