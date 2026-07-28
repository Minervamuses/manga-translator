from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_translator.benchmark.ocr_calibration import (
    CalibrationExample,
    train_calibration_artifact,
    write_calibration_artifact,
)
from manga_translator.ocr_confidence import (
    AcceptanceProfile,
    CalibrationArtifact,
    OCRConfidenceFeatures,
    decide_ocr_confidence,
    profile_for_text,
)


def _features(signal: float) -> OCRConfidenceFeatures:
    return OCRConfidenceFeatures(
        token_logprob=-8.0 + signal * 8.0,
        token_entropy=(1.0 - signal) * 8.0,
        token_margin=signal,
        detector_confidence=signal,
        mask_coverage=signal,
        candidate_disagreement=1.0 - signal,
        heuristic_quality=signal,
        crop_entropy=signal * 8.0,
        text_length=max(1, round(signal * 20)),
        view_stage=0 if signal > 0.5 else 2,
    )


def _synthetic_examples() -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    texts = ("一般對話內容", "漢", "BANG")
    for split_index, split in enumerate(("train", "dev", "test")):
        for index in range(60):
            signal = (index + 0.5) / 60
            profile_text = texts[index % len(texts)]
            examples.append(
                CalibrationExample(
                    example_id=f"{split}-{index}",
                    title=f"title-{split_index}",
                    split=split,
                    text=profile_text,
                    features=_features(signal),
                    correct=signal >= 0.55,
                    is_no_text=signal < 0.22,
                )
            )
    return examples


def test_untrained_score_is_explicitly_heuristic_and_profiles_are_separate() -> None:
    dialogue = decide_ocr_confidence("一般對話內容", _features(0.6))
    short = decide_ocr_confidence("漢", _features(0.6))
    sfx = decide_ocr_confidence("BANG", _features(0.6))

    assert dialogue.confidence_kind == "heuristic"
    assert dialogue.no_text_probability is None
    assert profile_for_text("一般對話內容") is AcceptanceProfile.DIALOGUE
    assert short.profile is AcceptanceProfile.SHORT_CJK
    assert sfx.profile is AcceptanceProfile.LATIN_SFX
    assert short.threshold < dialogue.threshold
    assert sfx.threshold < dialogue.threshold


def test_eval_training_improves_heldout_brier_and_uses_dev_for_thresholds(tmp_path: Path) -> None:
    artifact = train_calibration_artifact(
        _synthetic_examples(), model_revision="abc1234", preprocess_version="v1"
    )
    metrics = artifact.heldout_metrics

    assert metrics["calibrated_brier"] < metrics["uncalibrated_brier"]
    assert metrics["threshold_selection_split"] == "dev"
    assert metrics["evaluation_split"] == "test"
    assert len(metrics["reliability"]) == 10
    assert set(metrics["profiles"]) == {profile.value for profile in AcceptanceProfile}

    path = tmp_path / "calibration.json"
    write_calibration_artifact(path, artifact)
    loaded = CalibrationArtifact.from_json(
        path,
        model_revision="abc1234",
        preprocess_version="v1",
        corpus_sha256=artifact.corpus_sha256,
    )
    assert loaded == artifact


def test_no_text_gate_rejects_hallucination_without_hurting_strong_short_or_sfx() -> None:
    artifact = train_calibration_artifact(
        _synthetic_examples(), model_revision="abc1234", preprocess_version="v1"
    )

    hallucination = decide_ocr_confidence("看似文字", _features(0.05), artifact)
    short = decide_ocr_confidence("漢", _features(0.9), artifact)
    sfx = decide_ocr_confidence("BANG", _features(0.9), artifact)

    assert not hallucination.accepted
    assert hallucination.no_text_probability is not None
    assert hallucination.no_text_probability >= artifact.no_text_threshold
    assert short.accepted
    assert sfx.accepted


@pytest.mark.parametrize("field", ["model_revision", "preprocess_version", "corpus_sha256"])
def test_calibration_fingerprint_mismatch_is_rejected(tmp_path: Path, field: str) -> None:
    artifact = train_calibration_artifact(
        _synthetic_examples(), model_revision="abc1234", preprocess_version="v1"
    )
    path = tmp_path / "calibration.json"
    write_calibration_artifact(path, artifact)
    expected = {
        "model_revision": "abc1234",
        "preprocess_version": "v1",
        "corpus_sha256": artifact.corpus_sha256,
    }
    expected[field] = "mismatch"

    with pytest.raises(ValueError, match=field):
        CalibrationArtifact.from_json(path, **expected)


def test_repository_calibration_manifest_does_not_fake_missing_corpus() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "assets/calibration/ocr/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked"
    assert manifest["artifact"] is None
