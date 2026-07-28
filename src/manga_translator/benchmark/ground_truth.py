"""Ground-truth import and validation for the v0.3.2 regression pages."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..baseline import sha256_file
from .overlays import generate_page_artifacts

SCHEMA_VERSION = "page_ground_truth.v1"
PROFILE_NAME = "regression_v032"
FIXTURE_TO_SOURCE = {
    "v032_caption_3_layout.json": "__#Uf008_#Ueff9#Ue7cc (3).jpg",
    "v032_caption_4_layout.json": "__#Uf008_#Ueff9#Ue7cc (4).jpg",
    "v032_caption_5_layout.json": "__#Uf008_#Ueff9#Ue7cc (5).jpg",
    "v032_dash_0211_layout.json": "0211_t_11takamatic006.jpg",
    "v032_overlap_0188_layout.json": "0188_ive_hwa002.jpg",
}
KNOWN_0188_CORRECTIONS = {
    (579, 1061, 92, 260): {
        "source": "私は…三番街自衛騎士団所属セシリー・キャンベルだ",
        "translation": "我是…第三街自衛騎士團所屬的塞西莉・坎貝爾",
    },
    (406, 1111, 146, 212): {
        "source": "男が酒場で騒いでいると通報を受けてこの始末ならどうなってこうなっただろう？",
        "translation": "接到有人在酒館鬧事的通報,結果怎麼會變成這樣?",
    },
}


@dataclass
class ValidationReport:
    pages: int = 0
    regions: int = 0
    unverified: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _crop_sha256(image: Image.Image, bbox: tuple[int, int, int, int]) -> str:
    x, y, width, height = bbox
    crop = image.crop((x, y, x + width, y + height))
    digest = hashlib.sha256()
    digest.update(crop.mode.encode("ascii"))
    digest.update(f"{crop.width}x{crop.height}".encode("ascii"))
    digest.update(crop.tobytes())
    return digest.hexdigest()


def _corrected_content(
    source_name: str, bbox: tuple[int, int, int, int], source: str, translation: str
) -> tuple[str, str, str | None, str | None, str]:
    correction = (
        KNOWN_0188_CORRECTIONS.get(bbox) if source_name == "0188_ive_hwa002.jpg" else None
    )
    if correction is None:
        return source, translation, None, None, "fixture_import"
    return (
        correction["source"],
        correction["translation"],
        "execution-spec",
        "2026-07-28",
        "P0-02-known-0188-swap",
    )


def build_page_ground_truth(root: Path, fixture_name: str) -> dict[str, Any]:
    source_name = FIXTURE_TO_SOURCE[fixture_name]
    source_relative = f"samples/before_fix/{source_name}"
    fixture_relative = f"validation_samples/{fixture_name}"
    source_path = root / source_relative
    fixture_path = root / fixture_relative
    groups = json.loads(fixture_path.read_text(encoding="utf-8"))
    page_sha256 = sha256_file(source_path)

    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        regions: list[dict[str, Any]] = []
        for order, group in enumerate(groups):
            bbox = tuple(int(value) for value in group["bbox"])
            source, translation, verified_by, verified_at, method = _corrected_content(
                source_name, bbox, group["source"], group["translation"]
            )
            region_key = f"{page_sha256[:12]}:{group['id']}"
            regions.append(
                {
                    "region_key": region_key,
                    "legacy_group_id": group["id"],
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                    },
                    "reading_order_index": order,
                    "source_text": {
                        "raw": source,
                        "nfc": unicodedata.normalize("NFC", source),
                    },
                    "fixed_translation": translation,
                    "source_crop_sha256": _crop_sha256(image, bbox),
                    "verified_by": verified_by,
                    "verified_at": verified_at,
                    "provenance": {
                        "method": method,
                        "source_path": fixture_relative,
                        "source_group_id": group["id"],
                    },
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "page_id": page_sha256,
        "page_sha256": page_sha256,
        "source_image": source_relative,
        "fixture_source": fixture_relative,
        "width": width,
        "height": height,
        "regions": regions,
    }


def prepare_profile(
    root: Path, profile: str = PROFILE_NAME, *, destination: Path | None = None
) -> list[Path]:
    if profile != PROFILE_NAME:
        raise ValueError(f"unknown benchmark profile: {profile}")
    destination = destination or root / "benchmarks" / profile
    pages_dir = destination / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fixture_name in FIXTURE_TO_SOURCE:
        page = build_page_ground_truth(root, fixture_name)
        slug = Path(page["source_image"]).stem
        page_path = pages_dir / f"{slug}.json"
        page_path.write_text(
            json.dumps(page, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(page_path)
        generate_page_artifacts(
            root / page["source_image"],
            page,
            destination,
            root / "fonts" / "NotoSansCJKtc-Regular.otf",
        )
    return written


def _validate_region(
    page: dict[str, Any],
    region: Any,
    index: int,
    seen: set[str],
    report: ValidationReport,
    image: Image.Image | None,
) -> None:
    location = f"{page.get('source_image', '<unknown>')} region[{index}]"
    if not isinstance(region, dict):
        report.errors.append(f"{location}: region must be an object")
        return
    key = region.get("region_key")
    if not isinstance(key, str) or not key:
        report.errors.append(f"{location}: missing region_key")
    elif key in seen:
        report.errors.append(f"{location}: duplicate region_key {key}")
    else:
        seen.add(key)

    bbox = region.get("bbox")
    if not isinstance(bbox, dict):
        report.errors.append(f"{location}: bbox must be an object")
        return
    values = [bbox.get(name) for name in ("x", "y", "width", "height")]
    if any(not isinstance(value, int) for value in values):
        report.errors.append(f"{location}: bbox values must be integers")
        return
    x, y, width, height = values
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        report.errors.append(f"{location}: invalid bbox")
    if x + width > page.get("width", 0) or y + height > page.get("height", 0):
        report.errors.append(f"{location}: bbox outside page")
    elif image is not None and region.get("source_crop_sha256") != _crop_sha256(
        image, (x, y, width, height)
    ):
        report.errors.append(f"{location}: source crop hash mismatch")

    if region.get("reading_order_index") != index:
        report.errors.append(f"{location}: reading_order_index mismatch")

    source_text = region.get("source_text")
    if not isinstance(source_text, dict) or not source_text.get("raw"):
        report.errors.append(f"{location}: missing source text")
    elif source_text.get("nfc") != unicodedata.normalize("NFC", source_text["raw"]):
        report.errors.append(f"{location}: source NFC mismatch")
    if not isinstance(region.get("fixed_translation"), str) or not region["fixed_translation"]:
        report.errors.append(f"{location}: missing fixed translation")
    if region.get("verified_by") is None:
        report.unverified += 1
    elif not isinstance(region.get("verified_by"), str) or not region["verified_by"]:
        report.errors.append(f"{location}: verified_by must be a non-empty string or null")
    elif not region.get("verified_at"):
        report.errors.append(f"{location}: verified_at required with verified_by")
    provenance = region.get("provenance")
    if not isinstance(provenance, dict) or any(
        not provenance.get(field) for field in ("method", "source_path", "source_group_id")
    ):
        report.errors.append(f"{location}: incomplete provenance")

    source_name = Path(str(page.get("source_image", ""))).name
    expected = (
        KNOWN_0188_CORRECTIONS.get((x, y, width, height))
        if source_name == "0188_ive_hwa002.jpg"
        else None
    )
    if expected and (
        source_text.get("raw") != expected["source"]
        or region.get("fixed_translation") != expected["translation"]
    ):
        report.errors.append(f"{location}: known 0188 anchor content mismatch")


def validate_profile(
    root: Path,
    profile: str = PROFILE_NAME,
    *,
    profile_dir: Path | None = None,
    require_verified: bool = False,
) -> ValidationReport:
    if profile != PROFILE_NAME:
        raise ValueError(f"unknown benchmark profile: {profile}")
    profile_dir = profile_dir or root / "benchmarks" / profile
    report = ValidationReport()
    pages = sorted((profile_dir / "pages").glob("*.json"))
    if len(pages) != len(FIXTURE_TO_SOURCE):
        report.errors.append(f"expected {len(FIXTURE_TO_SOURCE)} pages, found {len(pages)}")

    seen: set[str] = set()
    for path in pages:
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            report.errors.append(f"{path}: {error}")
            continue
        report.pages += 1
        if page.get("schema_version") != SCHEMA_VERSION:
            report.errors.append(f"{path}: unsupported schema_version")
        source_path = root / str(page.get("source_image", ""))
        page_image: Image.Image | None = None
        if not source_path.is_file():
            report.errors.append(f"{path}: source image missing")
        elif sha256_file(source_path) != page.get("page_sha256"):
            report.errors.append(f"{path}: page_sha256 mismatch")
        else:
            with Image.open(source_path) as opened:
                page_image = opened.convert("RGB")
            if page_image.size != (page.get("width"), page.get("height")):
                report.errors.append(f"{path}: page dimensions mismatch")
        if page.get("page_id") != page.get("page_sha256"):
            report.errors.append(f"{path}: page_id must equal page_sha256")
        regions = page.get("regions")
        if not isinstance(regions, list):
            report.errors.append(f"{path}: regions must be an array")
            continue
        report.regions += len(regions)
        for index, region in enumerate(regions):
            _validate_region(page, region, index, seen, report, page_image)

        labels_path = profile_dir / "labels" / f"{path.stem}.labels.json"
        if not labels_path.is_file():
            report.errors.append(f"{path}: labels artifact missing")
        else:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            expected_keys = [region.get("region_key") for region in regions]
            if labels.get("overlay_labels") != expected_keys:
                report.errors.append(f"{path}: overlay labels do not match region keys")
            if labels.get("contact_sheet_labels") != expected_keys:
                report.errors.append(f"{path}: contact sheet labels do not match region keys")

    if report.unverified:
        message = f"{report.unverified} regions await human verification"
        if require_verified:
            report.errors.append(message)
        else:
            report.warnings.append(message)
    return report
