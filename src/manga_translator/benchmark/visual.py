"""Visual-v1 artifact bundles, hard metrics, percentiles, and blind-review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ARTIFACT_NAMES = (
    "source_overlay",
    "safe_mask",
    "style_fingerprint",
    "shaped_runs",
    "layout_alpha",
    "inpainted_roi",
    "final_preview",
    "metrics",
)
ARTIFACT_FILES = {
    name: f"{name}.json" if name in {"style_fingerprint", "shaped_runs", "metrics"} else f"{name}.png"
    for name in ARTIFACT_NAMES
}
REVIEW_CRITERIA = (
    "readability",
    "font_size",
    "spacing",
    "position",
    "font_style",
    "color_stroke",
    "overall_preference",
)


@dataclass(frozen=True)
class VisualMetrics:
    page_id: str
    region_key: str
    mapping_complete: bool
    missing_glyphs: int
    clreq_hard_violations: int
    accepted_collisions: int
    outside_roi_changed_pixels: int
    alpha_containment: float
    font_size_ratio: float
    center_offset_px: float
    whitespace_ratio: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _distribution(records: list[VisualMetrics], field: str) -> dict[str, Any]:
    values = [float(getattr(record, field)) for record in records]
    if not records:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "worst": None}
    minimize = field in {"center_offset_px", "whitespace_ratio"}
    worst = max(records, key=lambda item: getattr(item, field)) if minimize else min(
        records, key=lambda item: getattr(item, field)
    )
    return {
        "p05": _percentile(values, 0.05),
        "p50": statistics.median(values) if values else 0.0,
        "p95": _percentile(values, 0.95),
        "worst": {
            "page_id": worst.page_id,
            "region_key": worst.region_key,
            "value": float(getattr(worst, field)),
        },
    }


def _manual_review_summary(
    reviews: Iterable[Mapping[str, Any]],
    *,
    eligible_region_keys: frozenset[str],
) -> dict[str, Any]:
    seen: set[str] = set()
    duplicate_keys: list[str] = []
    unknown_keys: list[str] = []
    invalid_keys: list[str] = []
    verified: list[Mapping[str, Any]] = []
    for row in reviews:
        region_key = str(row.get("region_key", ""))
        if not region_key:
            invalid_keys.append(region_key)
            continue
        if region_key in seen:
            duplicate_keys.append(region_key)
            continue
        seen.add(region_key)
        if region_key not in eligible_region_keys:
            unknown_keys.append(region_key)
            continue
        ratings = row.get("ratings")
        if not row.get("verified_by") or not isinstance(ratings, Mapping):
            continue
        valid = isinstance(row.get("critical_regression"), bool)
        for variant in ("legacy", "new"):
            values = ratings.get(variant)
            if not isinstance(values, Mapping):
                valid = False
                break
            for criterion in REVIEW_CRITERIA:
                rating = values.get(criterion)
                if (
                    isinstance(rating, bool)
                    or not isinstance(rating, (int, float))
                    or not 1 <= float(rating) <= 5
                ):
                    valid = False
                    break
        if valid:
            verified.append(row)
        else:
            invalid_keys.append(region_key)
    critical = [row.get("region_key") for row in verified if row.get("critical_regression")]
    comparisons: dict[str, dict[str, float]] = {}
    for criterion in REVIEW_CRITERIA:
        legacy = [float(row["ratings"]["legacy"][criterion]) for row in verified]
        new = [float(row["ratings"]["new"][criterion]) for row in verified]
        comparisons[criterion] = {
            "legacy_p50": statistics.median(legacy) if legacy else 0.0,
            "new_p50": statistics.median(new) if new else 0.0,
        }
    complete = (
        len(verified) >= 30
        and not duplicate_keys
        and not unknown_keys
        and not invalid_keys
    )
    no_regression = all(
        comparisons[name]["new_p50"] >= comparisons[name]["legacy_p50"]
        for name in ("readability", "overall_preference")
    )
    return {
        "status": "passed" if complete and not critical and no_regression else "blocked",
        "verified_groups": len(verified),
        "required_groups": 30,
        "critical_regressions": critical,
        "duplicate_region_keys": sorted(set(duplicate_keys)),
        "unknown_region_keys": sorted(set(unknown_keys)),
        "invalid_region_keys": sorted(set(invalid_keys)),
        "comparisons": comparisons,
    }


def resolve_blind_reviews(
    sheet: Mapping[str, Any],
    key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve completed A/B ratings without exposing variant identity in the sheet."""

    if sheet.get("schema_version") != "blind_review.v1":
        raise ValueError("unsupported blind-review sheet schema")
    if key.get("schema_version") != "blind_review_key.v1":
        raise ValueError("unsupported blind-review key schema")
    sheet_rows = sheet.get("rows")
    key_rows = key.get("rows")
    if not isinstance(sheet_rows, list) or not isinstance(key_rows, list):
        raise TypeError("blind-review rows must be lists")

    def unique_rows(rows: list[Any], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("blind_id"), str):
                raise TypeError(f"{label} rows must contain string blind_id values")
            blind_id = row["blind_id"]
            if blind_id in result:
                raise ValueError(f"duplicate blind_id in {label}: {blind_id}")
            result[blind_id] = row
        return result

    sheets = unique_rows(sheet_rows, "sheet")
    keys = unique_rows(key_rows, "key")
    if set(sheets) != set(keys):
        raise ValueError("blind-review sheet and key ID sets differ")

    resolved: list[dict[str, Any]] = []
    for blind_id in sorted(sheets):
        row = sheets[blind_id]
        new_variant = keys[blind_id].get("new_variant")
        if new_variant not in {"a", "b"}:
            raise ValueError(f"invalid new_variant for blind review {blind_id}")
        legacy_variant = "b" if new_variant == "a" else "a"
        reviewer = row.get("reviewer")
        criteria = row.get("criteria")
        critical = row.get("critical_regression")
        output: dict[str, Any] = {
            "region_key": row.get("region_key"),
            "verified_by": None,
            "critical_regression": critical,
        }
        if reviewer is None and critical is None:
            resolved.append(output)
            continue
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"blind review {blind_id} has an invalid reviewer")
        if not isinstance(critical, bool) or not isinstance(criteria, Mapping):
            raise TypeError(f"blind review {blind_id} is incomplete")
        ratings = {"legacy": {}, "new": {}}
        for criterion in REVIEW_CRITERIA:
            values = criteria.get(criterion)
            if not isinstance(values, Mapping):
                raise TypeError(f"blind review {blind_id} is missing {criterion}")
            for output_name, variant in (("legacy", legacy_variant), ("new", new_variant)):
                rating = values.get(variant)
                if (
                    isinstance(rating, bool)
                    or not isinstance(rating, (int, float))
                    or not 1 <= float(rating) <= 5
                ):
                    raise ValueError(
                        f"blind review {blind_id} has an invalid {criterion}/{variant} rating"
                    )
                ratings[output_name][criterion] = float(rating)
        output["verified_by"] = reviewer.strip()
        output["ratings"] = ratings
        resolved.append(output)
    return resolved


def build_visual_report(
    records: Iterable[VisualMetrics],
    *,
    reviews: Iterable[Mapping[str, Any]] = (),
    required_pages: int = 5,
) -> dict[str, Any]:
    rows = list(records)
    pages = {record.page_id for record in rows}
    region_keys = [record.region_key for record in rows]
    unique_region_keys = len(region_keys) == len(set(region_keys))
    hard = {
        "five_page_corpus": len(pages) >= required_pages,
        "mapping_100_percent": bool(rows)
        and unique_region_keys
        and all(record.mapping_complete for record in rows),
        "missing_glyphs_zero": bool(rows) and all(record.missing_glyphs == 0 for record in rows),
        "clreq_hard_violations_zero": bool(rows)
        and all(record.clreq_hard_violations == 0 for record in rows),
        "accepted_collisions_zero": bool(rows)
        and all(record.accepted_collisions == 0 for record in rows),
        "outside_roi_changes_zero": bool(rows)
        and all(record.outside_roi_changed_pixels == 0 for record in rows),
        "alpha_containment_at_least_0_995": bool(rows)
        and all(record.alpha_containment >= 0.995 for record in rows),
    }
    manual = _manual_review_summary(
        reviews,
        eligible_region_keys=frozenset(region_keys),
    )
    automated_status = "passed" if all(hard.values()) else "failed"
    return {
        "schema_version": "visual_report.v1",
        "status": "passed" if automated_status == "passed" and manual["status"] == "passed" else "blocked",
        "automated_status": automated_status,
        "page_count": len(pages),
        "group_count": len(rows),
        "hard_metrics": hard,
        "distributions": {
            field: _distribution(rows, field)
            for field in (
                "alpha_containment",
                "font_size_ratio",
                "center_offset_px",
                "whitespace_ratio",
            )
        },
        "manual_review": manual,
        "groups": [asdict(row) for row in rows],
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"could not encode visual artifact: {path.name}")
    path.write_bytes(encoded.tobytes())


def write_group_bundle(
    output_dir: Path,
    *,
    source_overlay: np.ndarray,
    safe_mask: np.ndarray,
    style_fingerprint: Mapping[str, Any],
    shaped_runs: Iterable[Mapping[str, Any]],
    layout_alpha: np.ndarray,
    inpainted_roi: np.ndarray,
    final_preview: np.ndarray,
    metrics: VisualMetrics,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "source_overlay": source_overlay,
        "safe_mask": safe_mask,
        "layout_alpha": layout_alpha,
        "inpainted_roi": inpainted_roi,
        "final_preview": final_preview,
    }
    paths: dict[str, str] = {}
    for name, image in images.items():
        path = output_dir / f"{name}.png"
        _write_image(path, image)
        paths[name] = path.as_posix()
    for name, payload in {
        "style_fingerprint": dict(style_fingerprint),
        "shaped_runs": list(shaped_runs),
        "metrics": asdict(metrics),
    }.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = path.as_posix()
    return paths


def build_blind_review_sheet(groups: Iterable[Mapping[str, Any]]) -> tuple[dict, dict]:
    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda row: (str(row["page_id"]), str(row["region_key"]))):
        token = hashlib.sha256(
            f"{group['page_id']}:{group['region_key']}".encode()
        ).hexdigest()[:16]
        new_is_a = int(token[-1], 16) % 2 == 0
        legacy = str(group["legacy_preview"])
        new = str(group["new_preview"])
        sheet_rows.append(
            {
                "blind_id": token,
                "page_id": group["page_id"],
                "region_key": group["region_key"],
                "preview_a": new if new_is_a else legacy,
                "preview_b": legacy if new_is_a else new,
                "criteria": {name: {"a": None, "b": None} for name in REVIEW_CRITERIA},
                "critical_regression": None,
                "reviewer": None,
            }
        )
        key_rows.append({"blind_id": token, "new_variant": "a" if new_is_a else "b"})
    return (
        {"schema_version": "blind_review.v1", "rows": sheet_rows},
        {"schema_version": "blind_review_key.v1", "rows": key_rows},
    )


def bootstrap_visual_v1(root: Path) -> dict[str, Any]:
    corpus = root / "benchmarks" / "regression_v032" / "pages"
    output = root / "benchmarks" / "visual_v1"
    output.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    blind_groups: list[dict[str, Any]] = []
    for path in sorted(corpus.glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        page_id = str(page["page_id"])
        expected = {name: f"pages/{page_id}/{filename}" for name, filename in ARTIFACT_FILES.items()}
        pages.append(
            {
                "page_id": page_id,
                "source_image": page["source_image"],
                "groups": len(page["regions"]),
                "verified_groups": sum(bool(item.get("verified_by")) for item in page["regions"]),
                "artifact_status": "pending_raqm_run",
                "expected_artifacts": expected,
            }
        )
        legacy_preview = f"benchmarks/regression_v032/overlays/{path.stem}.overlay.jpg"
        for region in page["regions"]:
            blind_groups.append(
                {
                    "page_id": page_id,
                    "region_key": region["region_key"],
                    "legacy_preview": legacy_preview,
                    "new_preview": f"benchmarks/visual_v1/pages/{page_id}/final_preview.png",
                }
            )
    sheet, key = build_blind_review_sheet(blind_groups)
    (output / "blind_review_sheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "blind_review_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_visual_report((), required_pages=len(pages))
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "visual_v1.manifest",
        "status": "blocked",
        "page_count": len(pages),
        "group_count": len(blind_groups),
        "pages": pages,
        "blockers": [
            "local_pillow_raqm_unavailable",
            "new_visual_artifacts_not_generated",
            "manual_blind_review_0_of_30",
        ],
        "engine_switch": "not_performed",
        "report": "report.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = bootstrap_visual_v1(args.root.resolve())
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
