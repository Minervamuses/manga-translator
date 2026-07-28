"""Generate Unicode-labelled overlays and contact sheets for human review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), size)


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90, subsampling=0)


def _preview_lines(text: str, *, width: int = 24, limit: int = 2) -> str:
    lines = [text[index : index + width] for index in range(0, len(text), width)] or [""]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1][:-1] + "…"
    return "\n".join(lines)


def generate_page_artifacts(
    image_path: Path,
    page: dict[str, Any],
    destination: Path,
    font_path: Path,
) -> dict[str, Path]:
    slug = image_path.stem
    labels = [region["region_key"] for region in page["regions"]]
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        label_font = _font(font_path, max(16, min(30, source.width // 80)))
        for region in page["regions"]:
            bbox = region["bbox"]
            x, y = bbox["x"], bbox["y"]
            right, bottom = x + bbox["width"], y + bbox["height"]
            draw.rectangle((x, y, right, bottom), outline=(255, 40, 40), width=4)
            text = region["region_key"]
            text_bbox = draw.textbbox((x, y), text, font=label_font)
            label_height = text_bbox[3] - text_bbox[1] + 6
            label_width = text_bbox[2] - text_bbox[0] + 8
            label_y = max(0, y - label_height)
            label_x = max(0, min(x, source.width - label_width))
            draw.rectangle(
                (
                    label_x,
                    label_y,
                    label_x + label_width,
                    label_y + label_height,
                ),
                fill=(0, 0, 0),
            )
            draw.text(
                (label_x + 4, label_y + 2), text, font=label_font, fill=(255, 255, 0)
            )

        overlay_path = destination / "overlays" / f"{slug}.overlay.jpg"
        _save_jpeg(overlay, overlay_path)

        card_width = 900
        card_height = 260
        sheet = Image.new("RGB", (card_width, card_height * len(page["regions"])), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        title_font = _font(font_path, 23)
        text_font = _font(font_path, 20)
        for index, region in enumerate(page["regions"]):
            bbox = region["bbox"]
            crop = source.crop(
                (
                    bbox["x"],
                    bbox["y"],
                    bbox["x"] + bbox["width"],
                    bbox["y"] + bbox["height"],
                )
            )
            crop.thumbnail((320, card_height - 24))
            top = index * card_height
            sheet.paste(crop, (12, top + 12))
            text_x = 350
            sheet_draw.text(
                (text_x, top + 12), region["region_key"], font=title_font, fill="black"
            )
            bbox_text = f"bbox: {bbox['x']},{bbox['y']},{bbox['width']},{bbox['height']}"
            sheet_draw.text(
                (text_x, top + 48), bbox_text, font=text_font, fill=(70, 70, 70)
            )
            source_text = region["source_text"]["raw"]
            translation = region["fixed_translation"]
            sheet_draw.multiline_text(
                (text_x, top + 82),
                _preview_lines(source_text),
                font=text_font,
                fill="black",
                spacing=3,
            )
            sheet_draw.multiline_text(
                (text_x, top + 140),
                _preview_lines(translation),
                font=text_font,
                fill=(0, 70, 130),
                spacing=3,
            )
            verification = region["verified_by"] or "UNVERIFIED"
            sheet_draw.text(
                (text_x, top + 208), verification, font=text_font, fill=(150, 30, 30)
            )
            sheet_draw.line(
                (0, top + card_height - 1, card_width, top + card_height - 1),
                fill=(180, 180, 180),
                width=1,
            )

        contact_path = destination / "contact_sheets" / f"{slug}.contact.jpg"
        _save_jpeg(sheet, contact_path)

    labels_path = destination / "labels" / f"{slug}.labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(
            {"overlay_labels": labels, "contact_sheet_labels": labels},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"overlay": overlay_path, "contact_sheet": contact_path, "labels": labels_path}
