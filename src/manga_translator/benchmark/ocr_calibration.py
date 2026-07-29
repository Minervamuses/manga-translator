"""Optional-eval training and held-out evaluation for OCR calibration artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..ocr_confidence import (
    DEFAULT_COEFFICIENTS,
    FEATURE_NAMES,
    AcceptanceProfile,
    CalibrationArtifact,
    OCRConfidenceFeatures,
    linear_score,
    profile_for_text,
)


@dataclass(frozen=True)
class CalibrationExample:
    example_id: str
    title: str
    split: Literal["train", "dev", "test"]
    text: str
    features: OCRConfidenceFeatures
    correct: bool
    is_no_text: bool


def _corpus_hash(examples: list[CalibrationExample]) -> str:
    payload = [
        {
            "example_id": item.example_id,
            "title": item.title,
            "split": item.split,
            "text": item.text,
            "features": item.features.__dict__,
            "correct": item.correct,
            "is_no_text": item.is_no_text,
        }
        for item in sorted(examples, key=lambda example: example.example_id)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _matrix(examples: list[CalibrationExample]) -> np.ndarray:
    return np.asarray(
        [[item.features.normalized()[name] for name in FEATURE_NAMES] for item in examples],
        dtype=np.float64,
    )


def _precision_recall_coverage(labels: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
    true_positive = int(np.count_nonzero(labels & accepted))
    false_positive = int(np.count_nonzero(~labels & accepted))
    positive = int(np.count_nonzero(labels))
    return {
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, positive),
        "coverage": float(np.mean(accepted)) if accepted.size else 0.0,
    }


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best = (float("-inf"), 1.0)
    for threshold in np.linspace(0.05, 0.95, 91):
        accepted = scores >= threshold
        metrics = _precision_recall_coverage(labels, accepted)
        objective = 2 * metrics["precision"] * metrics["recall"] / max(
            1e-9, metrics["precision"] + metrics["recall"]
        )
        candidate = (objective, -float(threshold))
        best = max(best, candidate)
    return -best[1]


def _reliability(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float | int]]:
    bins: list[dict[str, float | int]] = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(np.count_nonzero(selected)),
                "predicted": float(np.mean(scores[selected])) if np.any(selected) else 0.0,
                "observed": float(np.mean(labels[selected])) if np.any(selected) else 0.0,
            }
        )
    return bins


def train_calibration_artifact(
    examples: list[CalibrationExample],
    *,
    model_revision: str,
    preprocess_version: str,
) -> CalibrationArtifact:
    """Train on train, tune thresholds on dev, and report only on test."""

    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - depends on optional eval extra
        raise RuntimeError("install the eval extra to train OCR calibration") from error
    splits = {
        name: [item for item in examples if item.split == name]
        for name in ("train", "dev", "test")
    }
    if any(not rows for rows in splits.values()):
        raise ValueError("calibration requires non-empty train/dev/test splits")
    example_ids = [item.example_id for item in examples]
    if any(not example_id for example_id in example_ids) or len(set(example_ids)) != len(
        example_ids
    ):
        raise ValueError("calibration example_id values must be non-empty and unique")
    title_splits: dict[str, set[str]] = {}
    for item in examples:
        if not item.title:
            raise ValueError("calibration title must not be empty")
        title_splits.setdefault(item.title, set()).add(item.split)
    leaked_titles = sorted(title for title, assigned in title_splits.items() if len(assigned) > 1)
    if leaked_titles:
        raise ValueError(f"calibration titles leak across splits: {', '.join(leaked_titles)}")
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    if len({item.correct for item in train}) < 2:
        raise ValueError("calibration train split requires correct and incorrect examples")
    if len({item.is_no_text for item in train}) < 2:
        raise ValueError("calibration train split requires text and no-text examples")
    for split_name, rows in (("dev", dev), ("test", test)):
        if len({item.is_no_text for item in rows}) < 2:
            raise ValueError(f"calibration {split_name} split requires text and no-text examples")
        for profile in AcceptanceProfile:
            profile_rows = [item for item in rows if profile_for_text(item.text) is profile]
            if not profile_rows or len({item.correct for item in profile_rows}) < 2:
                raise ValueError(
                    f"calibration {split_name} split requires positive and negative "
                    f"{profile.value} examples"
                )
    train_raw = np.asarray([linear_score(item.features) for item in train])
    train_labels = np.asarray([item.correct for item in train], dtype=bool)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(train_raw, train_labels)

    no_text = LogisticRegression(random_state=0, max_iter=500).fit(
        _matrix(train), np.asarray([item.is_no_text for item in train], dtype=bool)
    )
    dev_no_text_probability = no_text.predict_proba(_matrix(dev))[:, 1]
    no_text_labels = np.asarray([item.is_no_text for item in dev], dtype=bool)
    no_text_threshold = _select_threshold(no_text_labels, dev_no_text_probability)

    dev_raw = np.asarray([linear_score(item.features) for item in dev])
    dev_calibrated = isotonic.predict(dev_raw)
    profile_thresholds: dict[str, float] = {}
    for profile in AcceptanceProfile:
        selected = np.asarray([profile_for_text(item.text) is profile for item in dev])
        profile_thresholds[profile.value] = _select_threshold(
            np.asarray([item.correct for item in dev], dtype=bool)[selected],
            dev_calibrated[selected],
        )

    test_raw = np.asarray([linear_score(item.features) for item in test])
    test_calibrated = isotonic.predict(test_raw)
    test_labels = np.asarray([item.correct for item in test], dtype=bool)
    test_no_text_probability = no_text.predict_proba(_matrix(test))[:, 1]
    no_text_accepted = test_no_text_probability >= no_text_threshold
    heldout = {
        "threshold_selection_split": "dev",
        "evaluation_split": "test",
        "uncalibrated_brier": float(np.mean((test_raw - test_labels) ** 2)),
        "calibrated_brier": float(np.mean((test_calibrated - test_labels) ** 2)),
        "reliability": _reliability(test_labels, test_calibrated),
        "no_text": _precision_recall_coverage(
            np.asarray([item.is_no_text for item in test], dtype=bool), no_text_accepted
        ),
        "profiles": {},
    }
    for profile in AcceptanceProfile:
        selected = np.asarray([profile_for_text(item.text) is profile for item in test])
        heldout["profiles"][profile.value] = _precision_recall_coverage(
            test_labels[selected], test_calibrated[selected] >= profile_thresholds[profile.value]
        )
    return CalibrationArtifact(
        schema_version="ocr_calibration.v1",
        model_revision=model_revision,
        preprocess_version=preprocess_version,
        corpus_sha256=_corpus_hash(examples),
        coefficients=dict(DEFAULT_COEFFICIENTS),
        intercept=0.38,
        isotonic_x=tuple(float(value) for value in isotonic.X_thresholds_),
        isotonic_y=tuple(float(value) for value in isotonic.y_thresholds_),
        no_text_coefficients={
            name: float(value) for name, value in zip(FEATURE_NAMES, no_text.coef_[0])
        },
        no_text_intercept=float(no_text.intercept_[0]),
        no_text_threshold=no_text_threshold,
        profile_thresholds=profile_thresholds,
        heldout_metrics=heldout,
    )


def write_calibration_artifact(path: Path, artifact: CalibrationArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), allow_nan=False, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
