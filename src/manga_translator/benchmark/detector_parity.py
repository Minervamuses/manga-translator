"""Fail-closed real FP32/FP16 detector parity evidence runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import detector as detector_module
from ..config import AppConfig, DetectionConfig, PostprocessConfig
from ..detector import DetectionResult, TextRegion, detect_text_regions
from ..image_io import read_image


class DetectorParityBlocked(RuntimeError):
    """Raised before an evidence file is touched when a real run is unavailable."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(blockers)
        super().__init__("detector parity blocked: " + "; ".join(self.blockers))


DetectorRunner = Callable[
    [list[tuple[Path, np.ndarray]], DetectionConfig, PostprocessConfig, bool],
    list[DetectionResult],
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_profile_pages(root: Path, profile: str) -> list[Path]:
    pages: list[Path] = []
    for record_path in sorted((root / "benchmarks" / profile / "pages").glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            source = (root / str(record["source_image"])).resolve()
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if source.is_file():
            pages.append(source)
    return pages


def _box_iou(left: TextRegion, right: TextRegion) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def _best_match(region: TextRegion, candidates: list[TextRegion]) -> tuple[float, TextRegion | None]:
    if not candidates:
        return 0.0, None
    matched = max(candidates, key=lambda candidate: _box_iou(region, candidate))
    return _box_iou(region, matched), matched


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_on = left > 0
    right_on = right > 0
    union = int(np.count_nonzero(left_on | right_on))
    return 1.0 if union == 0 else int(np.count_nonzero(left_on & right_on)) / union


def _default_runner(
    pages: list[tuple[Path, np.ndarray]],
    detection: DetectionConfig,
    postprocess: PostprocessConfig,
    half: bool,
) -> list[DetectionResult]:
    detector_module._detector = None
    detector_module._detector_key = None
    current = detection.model_copy(update={"device": "cuda", "half": half})
    outputs: list[DetectionResult] = []
    for path, image in pages:
        result = detect_text_regions(image, current, postprocess)
        if result.issues:
            codes = ",".join(issue.code for issue in result.issues)
            raise DetectorParityBlocked((f"detector_runtime_fallback:{path.name}:{codes}",))
        loaded = detector_module._detector
        if loaded is None or bool(getattr(loaded, "half", None)) is not half:
            raise DetectorParityBlocked((f"detector_precision_not_applied:{path.name}",))
        outputs.append(result)
    return outputs


def _metrics(
    pages: list[tuple[Path, np.ndarray]],
    fp32_results: list[DetectionResult],
    fp16_results: list[DetectionResult],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    box_ious: list[float] = []
    score_errors: list[float] = []
    mask_ious: list[float] = []
    page_metrics: list[dict[str, Any]] = []
    small_total = small_matched = fp32_count = fp16_count = 0
    for (path, image), fp32, fp16 in zip(pages, fp32_results, fp16_results, strict=True):
        page_box_ious: list[float] = []
        page_score_errors: list[float] = []
        page_area = image.shape[0] * image.shape[1]
        fp32_count += len(fp32.regions_post)
        fp16_count += len(fp16.regions_post)
        page_mask_iou = _mask_iou(fp32.mask, fp16.mask)
        mask_ious.append(page_mask_iou)
        for region in fp32.regions_post:
            overlap, matched = _best_match(region, fp16.regions_post)
            box_ious.append(overlap)
            page_box_ious.append(overlap)
            if matched is not None:
                error = abs(region.confidence - matched.confidence)
                score_errors.append(error)
                page_score_errors.append(error)
            if region.area / max(1, page_area) <= thresholds["small_text_max_area_ratio"]:
                small_total += 1
                small_matched += overlap >= 0.5
        page_metrics.append(
            {
                "source": path.name,
                "source_sha256": _sha256_file(path),
                "fp32_boxes": len(fp32.regions_post),
                "fp16_boxes": len(fp16.regions_post),
                "box_mean_iou": float(np.mean(page_box_ious)) if page_box_ious else 1.0,
                "matched_score_mae": (
                    float(np.mean(page_score_errors)) if page_score_errors else 0.0
                ),
                "mask_iou": page_mask_iou,
            }
        )
    aggregate = {
        "box_count_ratio": min(fp32_count, fp16_count) / max(fp32_count, fp16_count, 1),
        "box_mean_iou": float(np.mean(box_ious)) if box_ious else 1.0,
        "matched_score_mae": float(np.mean(score_errors)) if score_errors else 0.0,
        "mask_iou": float(np.mean(mask_ious)) if mask_ious else 1.0,
        "small_text_recall": small_matched / small_total if small_total else 1.0,
        "fp32_box_count": fp32_count,
        "fp16_box_count": fp16_count,
        "small_text_count": small_total,
    }
    return aggregate, page_metrics


def run_detector_parity(
    root: Path,
    *,
    profile: str = "regression_v032",
    output: Path | None = None,
    runner: DetectorRunner | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run both real precisions and atomically publish evidence only when complete."""

    root = root.resolve()
    output = output or Path("benchmarks/detector_fp16_parity.json")
    output = output if output.is_absolute() else root / output
    previous = json.loads(output.read_text(encoding="utf-8"))
    required_pages = int(previous.get("required_pages", 5))
    thresholds = {key: float(value) for key, value in previous["thresholds"].items()}
    config = AppConfig.from_yaml(root / "config.yaml")
    paths = _load_profile_pages(root, profile)
    blockers: list[str] = []
    if runner is None and not torch.cuda.is_available():
        blockers.append("target_cuda_unavailable")
    if not config.detection.model_path.is_file():
        blockers.append("detector_model_missing")
    if len(paths) != required_pages:
        blockers.append(f"real_pages_required:{required_pages}:found:{len(paths)}")
    loaded_pages: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        image = read_image(path)
        if image is None:
            blockers.append(f"source_decode_failed:{path.name}")
        else:
            loaded_pages.append((path, image))
    if blockers:
        raise DetectorParityBlocked(blockers)

    actual_runner = runner or _default_runner
    fp32 = actual_runner(loaded_pages, config.detection, config.postprocess, False)
    fp16 = actual_runner(loaded_pages, config.detection, config.postprocess, True)
    if len(fp32) != required_pages or len(fp16) != required_pages:
        raise DetectorParityBlocked(("runner_returned_incomplete_page_set",))
    aggregate, page_metrics = _metrics(loaded_pages, fp32, fp16, thresholds)
    checks = {
        "box_count_ratio": aggregate["box_count_ratio"] >= thresholds["box_count_ratio_min"],
        "box_mean_iou": aggregate["box_mean_iou"] >= thresholds["box_mean_iou_min"],
        "matched_score_mae": (
            aggregate["matched_score_mae"] <= thresholds["matched_score_mae_max"]
        ),
        "mask_iou": aggregate["mask_iou"] >= thresholds["mask_iou_min"],
        "small_text_recall": (
            aggregate["small_text_recall"] >= thresholds["small_text_recall_min"]
        ),
    }
    evidence = {
        "schema_version": "detector_fp16_parity.v2",
        "status": "passed" if all(checks.values()) else "failed",
        "run_id": f"detector-parity-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "required_pages": required_pages,
        "mock": False,
        "source": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "write-tree"),
        },
        "corpus_sha256": _canonical_sha256(
            [{"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)} for path, _ in loaded_pages]
        ),
        "model": {
            "path": config.detection.model_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(config.detection.model_path),
        },
        "config_sha256": _canonical_sha256(
            {
                "detection": config.detection.model_dump(mode="json"),
                "postprocess": config.postprocess.model_dump(mode="json"),
            }
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "injected",
        },
        "thresholds": thresholds,
        "checks": checks,
        "metrics": aggregate,
        "pages": page_metrics,
    }
    _atomic_write_json(output, evidence)
    return output, evidence
