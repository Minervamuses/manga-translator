"""Ground-truth import and validation for the v0.3.2 regression pages."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..baseline import sha256_file
from .overlays import generate_page_artifacts

SCHEMA_VERSION = "page_ground_truth.v1"
PROFILE_NAME = "regression_v032"
SCHEMA_PATH = Path("benchmarks/schema/page_ground_truth.v1.json")
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


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _resolve_schema_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    target: Any = root_schema
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or key not in target:
            raise ValueError(f"unresolved schema reference: {reference}")
        target = target[key]
    if not isinstance(target, dict):
        raise TypeError(f"schema reference is not an object: {reference}")
    return target


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str,
) -> list[str]:
    if "$ref" in schema:
        schema = _resolve_schema_ref(root_schema, str(schema["$ref"]))

    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: must equal {schema['const']!r}")

    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if isinstance(expected_types, list) and not any(
        _matches_json_type(value, str(expected)) for expected in expected_types
    ):
        errors.append(f"{location}: expected type {' or '.join(map(str, expected_types))}")
        return errors

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{location}: missing required property {key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(properties)):
                errors.append(f"{location}: unexpected property {key}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                errors.extend(
                    _validate_schema_value(
                        value[key],
                        child_schema,
                        root_schema,
                        f"{location}.{key}",
                    )
                )

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{location}: requires at least {minimum_items} items")
        prefix_items = schema.get("prefixItems", [])
        for index, child_schema in enumerate(prefix_items):
            if index < len(value) and isinstance(child_schema, dict):
                errors.extend(
                    _validate_schema_value(
                        value[index], child_schema, root_schema, f"{location}[{index}]"
                    )
                )
        items = schema.get("items")
        if items is False and len(value) > len(prefix_items):
            errors.append(f"{location}: contains unexpected tuple items")
        elif isinstance(items, dict):
            start = len(prefix_items)
            for index, item in enumerate(value[start:], start=start):
                errors.extend(
                    _validate_schema_value(
                        item, items, root_schema, f"{location}[{index}]"
                    )
                )

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{location}: string is shorter than {minimum_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            errors.append(f"{location}: string does not match {pattern}")
        if schema.get("format") == "date":
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                errors.append(f"{location}: invalid ISO date")
            else:
                if parsed.isoformat() != value:
                    errors.append(f"{location}: invalid ISO date")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{location}: number must be finite")
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{location}: number is below minimum {minimum}")
    return errors


def validate_page_schema(page: Any, schema: dict[str, Any]) -> list[str]:
    """Apply the checked-in page schema without an optional runtime dependency."""

    return _validate_schema_value(page, schema, schema, "page")


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
                    "page_sha256": page_sha256,
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


def _project_file(
    root: Path,
    relative: Any,
    *,
    location: str,
    report: ValidationReport,
) -> Path | None:
    if not isinstance(relative, str) or not relative:
        report.errors.append(f"{location}: project path must be a non-empty string")
        return None
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        report.errors.append(f"{location}: project path escapes repository")
        return None
    try:
        candidate = (root / candidate_relative).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        report.errors.append(f"{location}: invalid project path: {error}")
        return None
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        report.errors.append(f"{location}: project path escapes repository")
        return None
    return candidate


def _validate_region(
    root: Path,
    page: dict[str, Any],
    region: Any,
    index: int,
    seen: set[str],
    report: ValidationReport,
    image: Image.Image | None,
    fixture_groups: dict[str, dict[str, Any]],
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

    legacy_group_id = region.get("legacy_group_id")
    expected_key = f"{str(page.get('page_sha256', ''))[:12]}:{legacy_group_id}"
    if key != expected_key:
        report.errors.append(f"{location}: region_key does not match page/group identity")
    if region.get("page_sha256") != page.get("page_sha256"):
        report.errors.append(f"{location}: region page_sha256 mismatch")

    bbox = region.get("bbox")
    if not isinstance(bbox, dict):
        report.errors.append(f"{location}: bbox must be an object")
        return
    values = [bbox.get(name) for name in ("x", "y", "width", "height")]
    if any(not isinstance(value, int) for value in values):
        report.errors.append(f"{location}: bbox values must be integers")
        return
    x, y, width, height = values
    invalid_bbox = width <= 0 or height <= 0 or x < 0 or y < 0
    if invalid_bbox:
        report.errors.append(f"{location}: invalid bbox")
    page_width = page.get("width") if isinstance(page.get("width"), int) else 0
    page_height = page.get("height") if isinstance(page.get("height"), int) else 0
    if invalid_bbox:
        pass
    elif x + width > page_width or y + height > page_height:
        report.errors.append(f"{location}: bbox outside page")
    elif image is not None and region.get("source_crop_sha256") != _crop_sha256(
        image, (x, y, width, height)
    ):
        report.errors.append(f"{location}: source crop hash mismatch")

    polygon = region.get("polygon")
    if isinstance(polygon, list):
        for point_index, point in enumerate(polygon):
            if not isinstance(point, list) or len(point) != 2:
                continue
            point_x, point_y = point
            if not all(_is_finite_number(value) for value in point):
                continue
            if not (0 <= point_x <= page_width) or not (0 <= point_y <= page_height):
                report.errors.append(f"{location}: polygon[{point_index}] outside page")

    if region.get("reading_order_index") != index:
        report.errors.append(f"{location}: reading_order_index mismatch")

    source_text_value = region.get("source_text")
    source_text = source_text_value if isinstance(source_text_value, dict) else {}
    raw_source = source_text.get("raw")
    if not isinstance(raw_source, str) or not raw_source:
        report.errors.append(f"{location}: missing source text")
    elif source_text.get("nfc") != unicodedata.normalize("NFC", raw_source):
        report.errors.append(f"{location}: source NFC mismatch")
    if not isinstance(region.get("fixed_translation"), str) or not region["fixed_translation"]:
        report.errors.append(f"{location}: missing fixed translation")
    if region.get("verified_by") is None:
        report.unverified += 1
        if region.get("verified_at") is not None:
            report.errors.append(f"{location}: verified_at must be null without verified_by")
    elif not isinstance(region.get("verified_by"), str) or not region["verified_by"]:
        report.errors.append(f"{location}: verified_by must be a non-empty string or null")
    elif not region.get("verified_at"):
        report.errors.append(f"{location}: verified_at required with verified_by")
    provenance = region.get("provenance")
    if not isinstance(provenance, dict) or any(
        not provenance.get(field) for field in ("method", "source_path", "source_group_id")
    ):
        report.errors.append(f"{location}: incomplete provenance")
        provenance = {}
    if provenance.get("source_path") != page.get("fixture_source"):
        report.errors.append(f"{location}: provenance source_path must match fixture_source")
    if provenance.get("source_group_id") != legacy_group_id:
        report.errors.append(f"{location}: provenance source_group_id mismatch")

    fixture_group = fixture_groups.get(str(legacy_group_id))
    if fixture_group is None:
        report.errors.append(f"{location}: provenance source group missing from fixture")
    else:
        fixture_bbox = fixture_group.get("bbox")
        if fixture_bbox != [x, y, width, height]:
            report.errors.append(f"{location}: bbox does not match provenance fixture")
        source_name = Path(str(page.get("source_image", ""))).name
        correction = (
            KNOWN_0188_CORRECTIONS.get((x, y, width, height))
            if source_name == "0188_ive_hwa002.jpg"
            else None
        )
        expected_method = "P0-02-known-0188-swap" if correction else "fixture_import"
        if provenance.get("method") != expected_method:
            report.errors.append(f"{location}: provenance method mismatch")
        if correction is None and (
            source_text.get("raw") != fixture_group.get("source")
            or region.get("fixed_translation") != fixture_group.get("translation")
        ):
            report.errors.append(f"{location}: imported text does not match provenance fixture")
    provenance_path = _project_file(
        root,
        provenance.get("source_path"),
        location=f"{location} provenance.source_path",
        report=report,
    )
    if provenance_path is not None and not provenance_path.is_file():
        report.errors.append(f"{location}: provenance source_path missing")

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
    try:
        schema = json.loads((root / SCHEMA_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.errors.append(f"{SCHEMA_PATH}: schema unavailable: {error}")
        return report
    if not isinstance(schema, dict):
        report.errors.append(f"{SCHEMA_PATH}: schema root must be an object")
        return report
    pages = sorted((profile_dir / "pages").glob("*.json"))
    if len(pages) != len(FIXTURE_TO_SOURCE):
        report.errors.append(f"expected {len(FIXTURE_TO_SOURCE)} pages, found {len(pages)}")

    seen: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    expected_pairs = {
        (
            f"validation_samples/{fixture_name}",
            f"samples/before_fix/{source_name}",
        )
        for fixture_name, source_name in FIXTURE_TO_SOURCE.items()
    }
    for path in pages:
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            report.errors.append(f"{path}: {error}")
            continue
        report.pages += 1
        try:
            schema_errors = validate_page_schema(page, schema)
        except (TypeError, ValueError, re.error) as error:
            report.errors.append(f"{SCHEMA_PATH}: invalid schema: {error}")
            return report
        for schema_error in schema_errors:
            report.errors.append(f"{path}: schema: {schema_error}")
        if not isinstance(page, dict):
            continue
        if page.get("schema_version") != SCHEMA_VERSION:
            report.errors.append(f"{path}: unsupported schema_version")

        fixture_relative = page.get("fixture_source")
        source_relative = page.get("source_image")
        if isinstance(fixture_relative, str) and isinstance(source_relative, str):
            observed_pairs.add((fixture_relative, source_relative))
        expected_source = dict(expected_pairs).get(str(fixture_relative))
        if expected_source is None:
            report.errors.append(f"{path}: fixture_source is not part of {PROFILE_NAME}")
        elif source_relative != expected_source:
            report.errors.append(f"{path}: source_image does not match fixture_source")
        if isinstance(source_relative, str) and path.stem != Path(source_relative).stem:
            report.errors.append(f"{path}: page filename does not match source image")

        fixture_path = _project_file(
            root,
            fixture_relative,
            location=f"{path} fixture_source",
            report=report,
        )
        fixture_groups: dict[str, dict[str, Any]] = {}
        if fixture_path is not None:
            if not fixture_path.is_file():
                report.errors.append(f"{path}: fixture_source missing")
            else:
                try:
                    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    report.errors.append(f"{path}: fixture_source invalid: {error}")
                else:
                    if not isinstance(fixture_payload, list):
                        report.errors.append(f"{path}: fixture_source must contain an array")
                    else:
                        fixture_groups = {
                            str(group.get("id")): group
                            for group in fixture_payload
                            if isinstance(group, dict)
                        }

        source_path = _project_file(
            root,
            source_relative,
            location=f"{path} source_image",
            report=report,
        )
        page_image: Image.Image | None = None
        if source_path is None or not source_path.is_file():
            report.errors.append(f"{path}: source image missing")
        elif sha256_file(source_path) != page.get("page_sha256"):
            report.errors.append(f"{path}: page_sha256 mismatch")
        else:
            try:
                with Image.open(source_path) as opened:
                    page_image = opened.convert("RGB")
            except OSError as error:
                report.errors.append(f"{path}: source image unreadable: {error}")
            else:
                if page_image.size != (page.get("width"), page.get("height")):
                    report.errors.append(f"{path}: page dimensions mismatch")
        if page.get("page_id") != page.get("page_sha256"):
            report.errors.append(f"{path}: page_id must equal page_sha256")
        regions = page.get("regions")
        if not isinstance(regions, list):
            report.errors.append(f"{path}: regions must be an array")
            continue
        report.regions += len(regions)
        observed_group_ids: list[str] = []
        for index, region in enumerate(regions):
            if isinstance(region, dict) and isinstance(region.get("legacy_group_id"), str):
                observed_group_ids.append(region["legacy_group_id"])
            _validate_region(
                root,
                page,
                region,
                index,
                seen,
                report,
                page_image,
                fixture_groups,
            )
        expected_group_ids = set(fixture_groups)
        actual_group_ids = set(observed_group_ids)
        if actual_group_ids != expected_group_ids:
            missing = sorted(expected_group_ids - actual_group_ids)
            unexpected = sorted(actual_group_ids - expected_group_ids)
            report.errors.append(
                f"{path}: fixture group coverage mismatch: "
                f"missing={missing}; unexpected={unexpected}"
            )
        if len(observed_group_ids) != len(actual_group_ids):
            report.errors.append(f"{path}: duplicate legacy_group_id")

        labels_path = profile_dir / "labels" / f"{path.stem}.labels.json"
        if not labels_path.is_file():
            report.errors.append(f"{path}: labels artifact missing")
        else:
            try:
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                report.errors.append(f"{path}: labels artifact invalid: {error}")
                continue
            if not isinstance(labels, dict):
                report.errors.append(f"{path}: labels artifact must be an object")
                continue
            if any(not isinstance(region, dict) for region in regions):
                report.errors.append(f"{path}: labels cannot bind non-object regions")
                continue
            expected_keys = [region.get("region_key") for region in regions]
            if labels.get("overlay_labels") != expected_keys:
                report.errors.append(f"{path}: overlay labels do not match region keys")
            if labels.get("contact_sheet_labels") != expected_keys:
                report.errors.append(f"{path}: contact sheet labels do not match region keys")

    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - expected_pairs)
    if missing_pairs:
        report.errors.append(f"missing expected fixture/source pairs: {missing_pairs}")
    if unexpected_pairs:
        report.errors.append(f"unexpected fixture/source pairs: {unexpected_pairs}")
    if report.regions != 38:
        report.errors.append(f"expected 38 regions, found {report.regions}")

    if report.unverified:
        message = f"{report.unverified} regions await human verification"
        if require_verified:
            report.errors.append(message)
        else:
            report.warnings.append(message)
    return report
