from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_translator.benchmark.ocr import (
    CER_DENOMINATOR,
    NORMALIZATION_REVISION,
    OCRPrediction,
    evaluate_ocr_predictions,
    evaluate_ocr_switch_gate,
    validate_corpus_manifest,
)


def _manifest(*, count: int = 100) -> dict:
    categories = ("dialogue", "furigana", "short_cjk", "latin_sfx", "text_over_art")
    items = []
    for split_index, split in enumerate(("train", "dev", "test")):
        title = f"title-{split_index}"
        for index in range(count):
            category = categories[index % len(categories)]
            truth = "漢" if category == "short_cjk" else ("BANG" if category == "latin_sfx" else "日本語")
            items.append(
                {
                    "crop_id": f"{split}-text-{index}",
                    "title": title,
                    "split": split,
                    "path": f"{split}/text-{index}.png",
                    "is_text": True,
                    "truth": truth,
                    "category": category,
                    "direction": "vertical" if index % 2 else "horizontal",
                    "verified_by": "reviewer",
                }
            )
            items.append(
                {
                    "crop_id": f"{split}-art-{index}",
                    "title": title,
                    "split": split,
                    "path": f"{split}/art-{index}.png",
                    "is_text": False,
                    "truth": "",
                    "category": "no_text_art",
                    "direction": "none",
                    "verified_by": "reviewer",
                }
            )
    return {
        "normalization_revision": NORMALIZATION_REVISION,
        "cer_denominator": CER_DENOMINATOR,
        "items": items,
    }


def _predictions(items, *, latency: float, error: bool = False):
    predictions = []
    for item in items:
        text = item.truth if item.is_text else ""
        if error and item.is_text and item.category == "dialogue":
            text = "誤"
        predictions.append(
            OCRPrediction(item.crop_id, text, item.is_text, latency, peak_vram_mb=512.0)
        )
    return tuple(predictions)


def test_corpus_contract_requires_600_verified_crops_and_title_held_out_split() -> None:
    items = validate_corpus_manifest(_manifest())
    assert len(items) == 600

    too_small = _manifest(count=99)
    with pytest.raises(ValueError, match="300 text"):
        validate_corpus_manifest(too_small)

    leaked = _manifest()
    leaked["items"][0]["title"] = "title-1"
    with pytest.raises(ValueError, match="title leakage"):
        validate_corpus_manifest(leaked)


def test_metrics_fix_cer_denominator_and_cover_short_sfx_no_text_and_throughput() -> None:
    items = validate_corpus_manifest(_manifest())
    metrics = evaluate_ocr_predictions(items, _predictions(items, latency=2.0))

    assert metrics["mapping_100_percent"]
    assert metrics["normalized_cer"] == 0
    assert metrics["exact_match"] == 1
    assert metrics["short_cjk_retention"] == 1
    assert metrics["latin_sfx_retention"] == 1
    assert metrics["no_text_false_positive"] == 0
    assert metrics["accepted_output_cer"] == 0
    assert metrics["coverage"] == 1
    assert metrics["p50_latency_ms"] == 2
    assert metrics["p95_latency_ms"] == 2
    assert metrics["images_per_second"] == pytest.approx(500)
    assert metrics["peak_vram_mb"] == 512


def test_switch_gate_requires_quality_and_real_target_gpu_gain() -> None:
    items = validate_corpus_manifest(_manifest())
    baseline = _predictions(items, latency=4.0)
    faster = _predictions(items, latency=2.0)

    passed = evaluate_ocr_switch_gate(items, baseline, faster, target_gpu=True)
    unavailable_gpu = evaluate_ocr_switch_gate(items, baseline, faster, target_gpu=False)
    regressed = evaluate_ocr_switch_gate(
        items, baseline, _predictions(items, latency=2.0, error=True), target_gpu=True
    )

    assert passed["status"] == "passed"
    assert passed["orchestrator_switch"] == "allowed"
    assert unavailable_gpu["status"] == "blocked"
    assert unavailable_gpu["orchestrator_switch"] == "not_performed"
    assert regressed["status"] == "blocked"
    assert regressed["checks"]["no_significant_category_cer_regression"] is False


def test_repository_ocr_v1_manifest_truthfully_blocks_switch() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "benchmarks/ocr_v1/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked"
    assert manifest["counts"]["verified_text_crops"] == 0
    assert manifest["counts"]["verified_no_text_crops"] == 0
    assert manifest["orchestrator_switch"] == "not_performed"
