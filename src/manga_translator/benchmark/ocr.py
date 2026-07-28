"""Title-held-out OCR corpus contract, metrics, and switch gate."""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class OCRPrediction:
    crop_id: str
    text: str
    accepted: bool
    latency_ms: float
    peak_vram_mb: float = 0.0


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
    if manifest.get("normalization_revision") != NORMALIZATION_REVISION:
        raise ValueError("OCR corpus normalization revision mismatch")
    if manifest.get("cer_denominator") != CER_DENOMINATOR:
        raise ValueError("OCR corpus CER denominator mismatch")
    items = tuple(OCRCorpusItem(**item) for item in manifest.get("items", []))
    text = [item for item in items if item.is_text]
    no_text = [item for item in items if not item.is_text]
    if len(text) < 300 or len(no_text) < 300:
        raise ValueError("OCR corpus requires at least 300 text and 300 no-text crops")
    if any(not item.verified_by for item in items):
        raise ValueError("every OCR corpus crop must be human verified")
    titles_by_split: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.split not in {"train", "dev", "test"}:
            raise ValueError(f"invalid OCR corpus split: {item.split}")
        titles_by_split[item.split].add(item.title)
    if len(set().union(*titles_by_split.values())) < 3:
        raise ValueError("OCR corpus requires at least three titles")
    for left in ("train", "dev", "test"):
        for right in ("train", "dev", "test"):
            if left < right and titles_by_split[left] & titles_by_split[right]:
                raise ValueError("OCR corpus title leakage across splits")
    required_categories = TEXT_CATEGORIES - {item.category for item in text}
    if required_categories:
        raise ValueError(f"OCR corpus missing text categories: {sorted(required_categories)}")
    return items


def corpus_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile * 100))


def evaluate_ocr_predictions(
    items: tuple[OCRCorpusItem, ...],
    predictions: tuple[OCRPrediction, ...],
) -> dict[str, Any]:
    by_id = {prediction.crop_id: prediction for prediction in predictions}
    mapping_complete = len(by_id) == len(items) and set(by_id) == {item.crop_id for item in items}
    text_items = [item for item in items if item.is_text]
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
    no_text_items = [item for item in items if not item.is_text]
    no_text_fp = sum(
        bool(by_id.get(item.crop_id))
        and by_id[item.crop_id].accepted
        and bool(normalize_benchmark_text(by_id[item.crop_id].text))
        for item in no_text_items
    )
    latencies = [prediction.latency_ms for prediction in predictions]
    total_seconds = sum(latencies) / 1000.0
    return {
        "mapping_100_percent": mapping_complete,
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
        "images_per_second": len(predictions) / max(1e-9, total_seconds),
        "peak_vram_mb": max((prediction.peak_vram_mb for prediction in predictions), default=0.0),
    }


def _paired_cer_delta_interval(
    items: tuple[OCRCorpusItem, ...],
    baseline: dict[str, OCRPrediction],
    candidate: dict[str, OCRPrediction],
    category: str,
    *,
    samples: int = 1000,
) -> tuple[float, float]:
    relevant = [item for item in items if item.is_text and item.category == category]
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
    intervals = {
        category: _paired_cer_delta_interval(
            items, baseline_by_id, candidate_by_id, category
        )
        for category in sorted(TEXT_CATEGORIES)
    }
    significant_regression = {
        category: bounds[0] > 0 for category, bounds in intervals.items()
    }
    checks = {
        "mapping_100_percent": candidate["mapping_100_percent"],
        "no_significant_category_cer_regression": not any(significant_regression.values()),
        "no_text_fp_not_worse": candidate["no_text_false_positive"]
        <= baseline["no_text_false_positive"],
        "short_cjk_retention_not_worse": candidate["short_cjk_retention"]
        >= baseline["short_cjk_retention"],
        "latin_sfx_retention_not_worse": candidate["latin_sfx_retention"]
        >= baseline["latin_sfx_retention"],
        "target_gpu_batching_gain": target_gpu
        and candidate["images_per_second"] > baseline["images_per_second"] * 1.05,
    }
    return {
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "paired_cer_delta_95ci": intervals,
        "significant_regression": significant_regression,
        "orchestrator_switch": "allowed" if all(checks.values()) else "not_performed",
    }
