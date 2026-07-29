"""Title-held-out OCR corpus contract, metrics, and switch gate."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

NORMALIZATION_REVISION = "nfkc-strip-whitespace-v1"
CER_DENOMINATOR = "sum_normalized_ground_truth_codepoints_for_text_crops"
TEXT_CATEGORIES = frozenset({"dialogue", "furigana", "short_cjk", "latin_sfx", "text_over_art"})
SPLITS = ("train", "dev", "test")
DIRECTIONS = frozenset({"horizontal", "vertical", "rotated", "none"})


@dataclass(frozen=True)
class OCRCorpusItem:
    crop_id: str
    title: str
    split: str
    path: str
    is_text: bool
    truth: str
    category: str
    direction: str
    verified_by: str | None

    def __post_init__(self) -> None:
        for name, value in {
            "crop_id": self.crop_id,
            "title": self.title,
            "path": self.path,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"OCR corpus {name} must not be empty")
        if self.split not in SPLITS:
            raise ValueError(f"invalid OCR corpus split: {self.split}")
        if not isinstance(self.is_text, bool):
            raise TypeError("OCR corpus is_text must be boolean")
        if not isinstance(self.verified_by, str) or not self.verified_by.strip():
            raise ValueError("every OCR corpus crop must be human verified")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"invalid OCR corpus direction: {self.direction}")
        normalized_truth = normalize_benchmark_text(self.truth)
        if self.is_text:
            if self.category not in TEXT_CATEGORIES:
                raise ValueError(f"invalid OCR text category: {self.category}")
            if not normalized_truth:
                raise ValueError("OCR text crop truth must not be empty")
            if self.direction == "none":
                raise ValueError("OCR text crop direction must describe the text orientation")
        elif self.category != "no_text_art" or normalized_truth:
            raise ValueError("OCR no-text crop must use no_text_art with empty truth")


@dataclass(frozen=True)
class OCRPrediction:
    crop_id: str
    text: str
    accepted: bool
    latency_ms: float
    peak_vram_mb: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.crop_id, str) or not self.crop_id.strip():
            raise ValueError("OCR prediction crop_id must not be empty")
        if not isinstance(self.text, str):
            raise TypeError("OCR prediction text must be a string")
        if not isinstance(self.accepted, bool):
            raise TypeError("OCR prediction accepted must be boolean")
        for name, value in {
            "latency_ms": self.latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
        }.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"OCR prediction {name} must be finite and non-negative")


def normalize_benchmark_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").split())


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_char in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1]


def validate_corpus_manifest(manifest: dict[str, Any]) -> tuple[OCRCorpusItem, ...]:
    if manifest.get("schema_version") != "ocr_v1.manifest":
        raise ValueError("OCR corpus schema version mismatch")
    if manifest.get("normalization_revision") != NORMALIZATION_REVISION:
        raise ValueError("OCR corpus normalization revision mismatch")
    if manifest.get("cer_denominator") != CER_DENOMINATOR:
        raise ValueError("OCR corpus CER denominator mismatch")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list):
        raise TypeError("OCR corpus items must be an array")
    items = tuple(OCRCorpusItem(**item) for item in raw_items)
    crop_ids = tuple(item.crop_id for item in items)
    if len(set(crop_ids)) != len(crop_ids):
        raise ValueError("OCR corpus crop_id values must be unique")
    text = [item for item in items if item.is_text]
    no_text = [item for item in items if not item.is_text]
    titles = {item.title for item in items}
    minimums = manifest.get("minimums")
    if not isinstance(minimums, dict) or minimums.get("split_unit") != "title":
        raise ValueError("OCR corpus minimums must declare title-level splitting")
    required_text = int(minimums.get("verified_text_crops", 0))
    required_no_text = int(minimums.get("verified_no_text_crops", 0))
    required_titles = int(minimums.get("titles", 0))
    if required_text < 300 or required_no_text < 300 or required_titles < 3:
        raise ValueError("OCR corpus minimums cannot weaken the benchmark contract")
    if len(text) < required_text or len(no_text) < required_no_text:
        raise ValueError("OCR corpus requires at least 300 text and 300 no-text crops")
    if len(titles) < required_titles:
        raise ValueError("OCR corpus requires at least three titles")
    counts = manifest.get("counts")
    expected_counts = {
        "verified_text_crops": len(text),
        "verified_no_text_crops": len(no_text),
        "titles": len(titles),
    }
    if counts != expected_counts:
        raise ValueError("OCR corpus declared counts do not match items")
    titles_by_split: dict[str, set[str]] = defaultdict(set)
    for item in items:
        titles_by_split[item.split].add(item.title)
    for split in SPLITS:
        split_items = [item for item in items if item.split == split]
        if not titles_by_split[split] or not any(item.is_text for item in split_items) or not any(
            not item.is_text for item in split_items
        ):
            raise ValueError(f"OCR corpus {split} split requires a title, text, and no-text crops")
        missing_categories = TEXT_CATEGORIES - {
            item.category for item in split_items if item.is_text
        }
        if missing_categories:
            raise ValueError(
                f"OCR corpus {split} split missing text categories: {sorted(missing_categories)}"
            )
    for left in SPLITS:
        for right in SPLITS:
            if left < right and titles_by_split[left] & titles_by_split[right]:
                raise ValueError("OCR corpus title leakage across splits")
    return items


def corpus_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile * 100))


def evaluate_ocr_predictions(
    items: tuple[OCRCorpusItem, ...],
    predictions: tuple[OCRPrediction, ...],
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"invalid OCR evaluation split: {split}")
    evaluated_items = tuple(item for item in items if item.split == split)
    if not evaluated_items:
        raise ValueError(f"OCR evaluation split has no items: {split}")
    prediction_ids = tuple(prediction.crop_id for prediction in predictions)
    duplicate_ids = sorted(
        crop_id for crop_id in set(prediction_ids) if prediction_ids.count(crop_id) > 1
    )
    by_id: dict[str, OCRPrediction] = {}
    for prediction in predictions:
        by_id.setdefault(prediction.crop_id, prediction)
    expected_ids = {item.crop_id for item in evaluated_items}
    mapping_complete = not duplicate_ids and set(by_id) == expected_ids
    text_items = [item for item in evaluated_items if item.is_text]
    total_edits = total_reference = 0
    exact = accepted = accepted_edits = accepted_reference = 0
    short_total = short_retained = sfx_total = sfx_retained = 0
    category_values: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in text_items:
        prediction = by_id.get(item.crop_id, OCRPrediction(item.crop_id, "", False, 0.0))
        truth = normalize_benchmark_text(item.truth)
        output = normalize_benchmark_text(prediction.text)
        edits = _edit_distance(truth, output)
        denominator = len(truth)
        total_edits += edits
        total_reference += denominator
        exact += int(output == truth)
        category_values[item.category].append((edits, denominator))
        if prediction.accepted:
            accepted += 1
            accepted_edits += edits
            accepted_reference += denominator
        if item.category == "short_cjk":
            short_total += 1
            short_retained += int(prediction.accepted and output == truth)
        if item.category == "latin_sfx":
            sfx_total += 1
            sfx_retained += int(prediction.accepted and output == truth)
    no_text_items = [item for item in evaluated_items if not item.is_text]
    no_text_fp = sum(
        bool(by_id.get(item.crop_id))
        and by_id[item.crop_id].accepted
        and bool(normalize_benchmark_text(by_id[item.crop_id].text))
        for item in no_text_items
    )
    evaluated_predictions = [by_id[crop_id] for crop_id in expected_ids if crop_id in by_id]
    latencies = [prediction.latency_ms for prediction in evaluated_predictions]
    total_seconds = sum(latencies) / 1000.0
    return {
        "mapping_100_percent": mapping_complete,
        "evaluation_split": split,
        "duplicate_prediction_ids": duplicate_ids,
        "normalized_cer": total_edits / max(1, total_reference),
        "exact_match": exact / max(1, len(text_items)),
        "short_cjk_retention": short_retained / max(1, short_total),
        "latin_sfx_retention": sfx_retained / max(1, sfx_total),
        "no_text_false_positive": no_text_fp / max(1, len(no_text_items)),
        "accepted_output_cer": accepted_edits / max(1, accepted_reference),
        "coverage": accepted / max(1, len(text_items)),
        "category_cer": {
            category: sum(edits for edits, _length in values)
            / max(1, sum(length for _edits, length in values))
            for category, values in sorted(category_values.items())
        },
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "images_per_second": len(evaluated_predictions) / max(1e-9, total_seconds),
        "peak_vram_mb": max(
            (prediction.peak_vram_mb for prediction in evaluated_predictions), default=0.0
        ),
    }


def _paired_cer_delta_interval(
    items: tuple[OCRCorpusItem, ...],
    baseline: dict[str, OCRPrediction],
    candidate: dict[str, OCRPrediction],
    category: str,
    *,
    split: str = "test",
    samples: int = 1000,
) -> tuple[float, float]:
    relevant = [
        item
        for item in items
        if item.split == split and item.is_text and item.category == category
    ]
    if not relevant:
        return (0.0, 0.0)
    deltas = []
    rng = random.Random(0)
    for _ in range(samples):
        sampled = [rng.choice(relevant) for _ in relevant]
        base_edits = candidate_edits = denominator = 0
        for item in sampled:
            truth = normalize_benchmark_text(item.truth)
            denominator += len(truth)
            base_edits += _edit_distance(truth, normalize_benchmark_text(baseline[item.crop_id].text))
            candidate_edits += _edit_distance(
                truth, normalize_benchmark_text(candidate[item.crop_id].text)
            )
        deltas.append((candidate_edits - base_edits) / max(1, denominator))
    return (_percentile(deltas, 0.025), _percentile(deltas, 0.975))


def evaluate_ocr_switch_gate(
    items: tuple[OCRCorpusItem, ...],
    baseline_predictions: tuple[OCRPrediction, ...],
    candidate_predictions: tuple[OCRPrediction, ...],
    *,
    target_gpu: bool,
) -> dict[str, Any]:
    baseline = evaluate_ocr_predictions(items, baseline_predictions)
    candidate = evaluate_ocr_predictions(items, candidate_predictions)
    baseline_by_id = {item.crop_id: item for item in baseline_predictions}
    candidate_by_id = {item.crop_id: item for item in candidate_predictions}
    mappings_complete = baseline["mapping_100_percent"] and candidate["mapping_100_percent"]
    intervals = (
        {
            category: _paired_cer_delta_interval(
                items, baseline_by_id, candidate_by_id, category
            )
            for category in sorted(TEXT_CATEGORIES)
        }
        if mappings_complete
        else {}
    )
    significant_regression = {
        category: bounds[0] > 0 for category, bounds in intervals.items()
    }
    checks = {
        "baseline_mapping_100_percent": baseline["mapping_100_percent"],
        "mapping_100_percent": candidate["mapping_100_percent"],
        "no_significant_category_cer_regression": not any(significant_regression.values()),
        "no_text_fp_not_worse": candidate["no_text_false_positive"]
        <= baseline["no_text_false_positive"],
        "short_cjk_retention_not_worse": candidate["short_cjk_retention"]
        >= baseline["short_cjk_retention"],
        "latin_sfx_retention_not_worse": candidate["latin_sfx_retention"]
        >= baseline["latin_sfx_retention"],
        "target_gpu_batching_gain": mappings_complete
        and target_gpu
        and candidate["images_per_second"] > baseline["images_per_second"] * 1.05,
    }
    return {
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "paired_cer_delta_95ci": intervals,
        "significant_regression": significant_regression,
        "blockers": [name for name, passed in checks.items() if not passed],
        "orchestrator_switch": "allowed" if all(checks.values()) else "not_performed",
    }
