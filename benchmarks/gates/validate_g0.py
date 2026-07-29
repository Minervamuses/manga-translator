"""Fail-closed validator for the G0 correctness and real-baseline gate."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from manga_translator.benchmark.detector_parity import _load_profile_pages
from manga_translator.benchmark.ground_truth import validate_profile
from manga_translator.benchmark.performance import (
    REQUIRED_STAGES,
    _load_corpus,
    _redacted_config_artifact,
    _source_fingerprint,
)
from manga_translator.config import AppConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: JSON root must be an object")
    return payload


def _resolve_artifact(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a non-empty string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact escapes repository: {relative}")
    return path


def _load_referenced_artifact(
    root: Path,
    reference: object,
    label: str,
    errors: list[str],
) -> tuple[Path | None, dict[str, Any] | None]:
    if not isinstance(reference, dict):
        errors.append(f"{label}: reference missing")
        return None, None
    try:
        path = _resolve_artifact(root, reference.get("path"))
    except ValueError as error:
        errors.append(f"{label}: {error}")
        return None, None
    if not path.is_file():
        errors.append(f"{label}: artifact missing")
        return path, None
    actual_hash = _sha256_file(path)
    if reference.get("sha256") != actual_hash:
        errors.append(f"{label}: artifact hash mismatch")
    try:
        payload = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"{label}: invalid JSON ({type(error).__name__})")
        return path, None
    if reference.get("run_id") is not None and reference.get("run_id") != payload.get(
        "run_id"
    ):
        errors.append(f"{label}: run ID mismatch")
    return path, payload


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else True)


def _review_errors(root: Path, gate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    report = validate_profile(root, require_verified=True)
    if not report.ok or report.unverified:
        errors.append("ground truth: verified 38-region corpus required")
    page_dir = root / "benchmarks" / "regression_v032" / "pages"
    reviews: Counter[tuple[object, object]] = Counter()
    for page_path in sorted(page_dir.glob("*.json")):
        page = _read_json(page_path)
        for region in page.get("regions", []):
            if isinstance(region, dict):
                reviews[(region.get("verified_by"), region.get("verified_at"))] += 1
    declared = gate.get("reviewers")
    if not isinstance(declared, list) or not declared:
        errors.append("ground truth: reviewer declaration missing")
        return errors
    declared_reviews: Counter[tuple[object, object]] = Counter()
    declared_total = 0
    for reviewer in declared:
        if not isinstance(reviewer, dict):
            errors.append("ground truth: reviewer declaration invalid")
            continue
        count = reviewer.get("regions")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            errors.append("ground truth: reviewer region count invalid")
            continue
        declared_reviews[(reviewer.get("id"), reviewer.get("verified_at"))] += count
        declared_total += count
    if declared_total != 38 or declared_reviews != reviews:
        errors.append("ground truth: reviewer declaration does not match 38 regions")
    return errors


def _parity_errors(root: Path, gate: dict[str, Any], parity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reference = gate.get("artifacts", {}).get("detector_parity", {})
    if parity.get("schema_version") != "detector_fp16_parity.v2":
        errors.append("detector parity: unsupported schema")
    if parity.get("status") != "passed":
        errors.append("detector parity: status is not passed")
    if parity.get("mock") is not False:
        errors.append("detector parity: mock evidence cannot satisfy G0")
    if reference.get("run_id") != parity.get("run_id"):
        errors.append("detector parity: gate run ID mismatch")
    parity_references = {
        "source_commit": parity.get("source", {}).get("commit"),
        "corpus_sha256": parity.get("corpus_sha256"),
        "model_sha256": parity.get("model", {}).get("sha256"),
        "config_sha256": parity.get("config_sha256"),
    }
    if any(reference.get(name) != value for name, value in parity_references.items()):
        errors.append("detector parity: gate fingerprint reference mismatch")

    thresholds = parity.get("thresholds")
    metrics = parity.get("metrics")
    checks = parity.get("checks")
    if not all(isinstance(value, dict) for value in (thresholds, metrics, checks)):
        errors.append("detector parity: metrics, thresholds, or checks missing")
    else:
        comparisons = {
            "box_count_ratio": metrics.get("box_count_ratio", -math.inf)
            >= thresholds.get("box_count_ratio_min", math.inf),
            "box_mean_iou": metrics.get("box_mean_iou", -math.inf)
            >= thresholds.get("box_mean_iou_min", math.inf),
            "matched_score_mae": metrics.get("matched_score_mae", math.inf)
            <= thresholds.get("matched_score_mae_max", -math.inf),
            "mask_iou": metrics.get("mask_iou", -math.inf)
            >= thresholds.get("mask_iou_min", math.inf),
            "small_text_recall": metrics.get("small_text_recall", -math.inf)
            >= thresholds.get("small_text_recall_min", math.inf),
        }
        if not all(comparisons.values()) or any(checks.get(name) is not True for name in comparisons):
            errors.append("detector parity: one or more thresholds failed")

    config = AppConfig.from_yaml(root / "config.yaml")
    current_config = _canonical_sha256(
        {
            "detection": config.detection.model_dump(mode="json"),
            "postprocess": config.postprocess.model_dump(mode="json"),
        }
    )
    if parity.get("config_sha256") != current_config:
        errors.append("detector parity: stale config fingerprint")
    if not config.detection.model_path.is_file() or parity.get("model", {}).get(
        "sha256"
    ) != _sha256_file(config.detection.model_path):
        errors.append("detector parity: stale model fingerprint")
    pages = _load_profile_pages(root, "regression_v032")
    current_corpus = _canonical_sha256(
        [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
            for path in pages
        ]
    )
    if parity.get("corpus_sha256") != current_corpus:
        errors.append("detector parity: stale corpus fingerprint")
    return errors


def _performance_errors(
    root: Path,
    gate: dict[str, Any],
    performance: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    reference = gate.get("artifacts", {}).get("performance", {})
    if performance.get("schema_version") != "performance_baseline.v2":
        errors.append("performance: unsupported schema")
    if reference.get("run_id") != performance.get("run_id"):
        errors.append("performance: gate run ID mismatch")
    performance_references = {
        "source_commit": performance.get("source", {}).get("git_commit"),
        "source_fingerprint_sha256": performance.get("source", {})
        .get("source_fingerprint", {})
        .get("sha256"),
        "corpus_sha256": performance.get("corpus", {}).get("sha256"),
        "config_sha256": performance.get("artifacts", {})
        .get("effective_config", {})
        .get("sha256"),
    }
    if any(reference.get(name) != value for name, value in performance_references.items()):
        errors.append("performance: gate fingerprint reference mismatch")
    real = performance.get("real_run")
    if not isinstance(real, dict):
        return [*errors, "performance: real run missing"]
    if (
        real.get("status") != "passed"
        or real.get("authoritative") is not True
        or real.get("performance_claim_allowed") is not True
        or real.get("environment_kind") != "real"
        or real.get("measurement_kind") != "full_pipeline"
        or real.get("blockers") != []
    ):
        errors.append("performance: authoritative real run required")

    cold = real.get("cold")
    warmup = real.get("warmup")
    measured = real.get("measurements")
    if not isinstance(cold, list) or len(cold) != 5:
        errors.append("performance: exactly 5 cold samples required")
    if not isinstance(warmup, list) or len(warmup) != 10:
        errors.append("performance: exactly 10 warmup samples required")
    if not isinstance(measured, list) or len(measured) != 25:
        errors.append("performance: exactly 25 measured samples required")
    samples = [
        sample
        for collection in (cold, warmup, measured)
        if isinstance(collection, list)
        for sample in collection
        if isinstance(sample, dict)
    ]
    if len(samples) != 40 or len({sample.get("sample_id") for sample in samples}) != 40:
        errors.append("performance: sample identities are incomplete or duplicated")
    required_stages = set(REQUIRED_STAGES)
    for sample in samples:
        profiler = sample.get("profiler")
        spans = profiler.get("spans", []) if isinstance(profiler, dict) else []
        observed = {
            span.get("stage") for span in spans if isinstance(span, dict)
        }
        if (
            not isinstance(profiler, dict)
            or profiler.get("environment_kind") != "real"
            or not required_stages <= observed
        ):
            errors.append("performance: mock or incomplete sample found in real measurements")
            break

    summary = real.get("summary")
    required_summary = (
        "p50_wall_ms",
        "p95_wall_ms",
        "cpu_rss_peak_bytes",
        "cuda_peak_allocated_bytes",
        "api_p50_ms",
        "api_p95_ms",
    )
    if not isinstance(summary, dict) or any(
        not _is_finite_number(summary.get(field), positive=True)
        for field in required_summary
    ):
        errors.append("performance: required real metrics missing")
    elif not isinstance(summary.get("worst_page"), dict) or not summary["worst_page"].get(
        "page_id"
    ):
        errors.append("performance: worst-page metric missing")

    current_source = _source_fingerprint(root)
    recorded_source = performance.get("source", {}).get("source_fingerprint", {})
    if recorded_source.get("sha256") != current_source.get("sha256"):
        errors.append("performance: stale source fingerprint")
    _pages, current_corpus, validation = _load_corpus(root)
    if performance.get("corpus", {}).get("sha256") != current_corpus:
        errors.append("performance: stale corpus fingerprint")
    if validation.get("unverified") != 0:
        errors.append("performance: corpus reviewer debt remains")
    current_config, _configured = _redacted_config_artifact(root)
    if performance.get("artifacts", {}).get("effective_config", {}).get(
        "sha256"
    ) != current_config.get("sha256"):
        errors.append("performance: stale config fingerprint")

    manifest_reference = gate.get("artifacts", {}).get("performance_manifest", {})
    try:
        manifest_path = _resolve_artifact(root, manifest_reference.get("path"))
        manifest = _read_json(manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"performance manifest: invalid ({type(error).__name__})")
    else:
        if manifest_reference.get("sha256") != _sha256_file(manifest_path):
            errors.append("performance manifest: artifact hash mismatch")
        entry = next(
            (
                item
                for item in manifest.get("runs", [])
                if isinstance(item, dict) and item.get("run_id") == performance.get("run_id")
            ),
            None,
        )
        if (
            manifest.get("latest_run_id") != performance.get("run_id")
            or not isinstance(entry, dict)
            or entry.get("sha256") != reference.get("sha256")
            or entry.get("authoritative") is not True
            or entry.get("real_status") != "passed"
        ):
            errors.append("performance manifest: current authoritative entry mismatch")
    return errors


def validate_payloads(
    root: Path,
    gate: dict[str, Any],
    parity: dict[str, Any],
    performance: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if gate.get("schema_version") != "g0_correctness_baseline.v1":
        errors.append("gate: unsupported schema")
    if gate.get("gate") != "G0" or gate.get("status") != "passed":
        errors.append("gate: G0 passed status required")
    if gate.get("blockers") != []:
        errors.append("gate: blockers must be empty")
    errors.extend(_review_errors(root, gate))
    errors.extend(_parity_errors(root, gate, parity))
    errors.extend(_performance_errors(root, gate, performance))
    return errors


def validate_g0(root: Path, gate_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    gate_path = gate_path or root / "benchmarks" / "gates" / "g0_correctness_baseline.json"
    gate = _read_json(gate_path)
    errors: list[str] = []
    _parity_path, parity = _load_referenced_artifact(
        root,
        gate.get("artifacts", {}).get("detector_parity"),
        "detector parity",
        errors,
    )
    _performance_path, performance = _load_referenced_artifact(
        root,
        gate.get("artifacts", {}).get("performance"),
        "performance",
        errors,
    )
    if parity is not None and performance is not None:
        errors.extend(validate_payloads(root, gate, parity, performance))
    return {
        "gate": "G0",
        "status": "passed" if not errors else "blocked",
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path.cwd()
    report = validate_g0(root)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
