"""Fingerprint-bound OCR confidence calibration and acceptance profiles."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .text import grapheme_clusters

FEATURE_NAMES = (
    "token_logprob",
    "token_entropy",
    "token_margin",
    "detector_confidence",
    "mask_coverage",
    "candidate_disagreement",
    "heuristic_quality",
    "crop_entropy",
    "text_length",
    "view_stage",
)
DEFAULT_COEFFICIENTS = {
    "token_logprob": 0.22,
    "token_entropy": -0.08,
    "token_margin": 0.16,
    "detector_confidence": 0.12,
    "mask_coverage": 0.08,
    "candidate_disagreement": -0.14,
    "heuristic_quality": 0.24,
    "crop_entropy": 0.04,
    "text_length": 0.06,
    "view_stage": -0.04,
}
IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class AcceptanceProfile(StrEnum):
    DIALOGUE = "dialogue"
    SHORT_CJK = "short_cjk"
    LATIN_SFX = "latin_sfx"


@dataclass(frozen=True)
class OCRConfidenceFeatures:
    token_logprob: float
    token_entropy: float
    token_margin: float
    detector_confidence: float
    mask_coverage: float
    candidate_disagreement: float
    heuristic_quality: float
    crop_entropy: float
    text_length: int
    view_stage: int

    def __post_init__(self) -> None:
        numeric = {
            "token_logprob": self.token_logprob,
            "token_entropy": self.token_entropy,
            "token_margin": self.token_margin,
            "detector_confidence": self.detector_confidence,
            "mask_coverage": self.mask_coverage,
            "candidate_disagreement": self.candidate_disagreement,
            "heuristic_quality": self.heuristic_quality,
            "crop_entropy": self.crop_entropy,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.token_logprob > 0:
            raise ValueError("token_logprob must not be positive")
        if self.token_entropy < 0 or self.crop_entropy < 0:
            raise ValueError("entropy features must not be negative")
        for name in (
            "token_margin",
            "detector_confidence",
            "mask_coverage",
            "candidate_disagreement",
            "heuristic_quality",
        ):
            if not 0.0 <= float(numeric[name]) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name, value in {
            "text_length": self.text_length,
            "view_stage": self.view_stage,
        }.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def normalized(self) -> dict[str, float]:
        return {
            "token_logprob": float(np.clip((self.token_logprob + 8.0) / 8.0, 0, 1)),
            "token_entropy": float(np.clip(self.token_entropy / 8.0, 0, 1)),
            "token_margin": float(np.clip(self.token_margin, 0, 1)),
            "detector_confidence": float(np.clip(self.detector_confidence, 0, 1)),
            "mask_coverage": float(np.clip(self.mask_coverage, 0, 1)),
            "candidate_disagreement": float(np.clip(self.candidate_disagreement, 0, 1)),
            "heuristic_quality": float(np.clip(self.heuristic_quality, 0, 1)),
            "crop_entropy": float(np.clip(self.crop_entropy / 8.0, 0, 1)),
            "text_length": float(np.clip(self.text_length / 20.0, 0, 1)),
            "view_stage": float(np.clip(self.view_stage / 3.0, 0, 1)),
        }


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: str
    model_revision: str
    preprocess_version: str
    corpus_sha256: str
    coefficients: dict[str, float]
    intercept: float
    isotonic_x: tuple[float, ...]
    isotonic_y: tuple[float, ...]
    no_text_coefficients: dict[str, float]
    no_text_intercept: float
    no_text_threshold: float
    profile_thresholds: dict[str, float]
    heldout_metrics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != "ocr_calibration.v1":
            raise ValueError("unsupported OCR calibration schema_version")
        if IMMUTABLE_REVISION_PATTERN.fullmatch(self.model_revision) is None:
            raise ValueError("calibration model_revision must be a full immutable commit hash")
        if not self.preprocess_version.strip():
            raise ValueError("calibration preprocess_version must not be empty")
        if SHA256_PATTERN.fullmatch(self.corpus_sha256) is None:
            raise ValueError("calibration corpus_sha256 must be a SHA-256 digest")
        expected_features = set(FEATURE_NAMES)
        if set(self.coefficients) != expected_features:
            raise ValueError("calibration coefficients must cover exactly the supported features")
        if set(self.no_text_coefficients) != expected_features:
            raise ValueError(
                "calibration no_text_coefficients must cover exactly the supported features"
            )
        expected_profiles = {profile.value for profile in AcceptanceProfile}
        if set(self.profile_thresholds) != expected_profiles:
            raise ValueError("calibration profile_thresholds must cover every profile")
        finite_values = (
            *self.coefficients.values(),
            self.intercept,
            *self.no_text_coefficients.values(),
            self.no_text_intercept,
            self.no_text_threshold,
            *self.profile_thresholds.values(),
            *self.isotonic_x,
            *self.isotonic_y,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in finite_values):
            raise ValueError("calibration numeric values must be finite")
        if not 0.0 <= self.no_text_threshold <= 1.0 or any(
            not 0.0 <= threshold <= 1.0 for threshold in self.profile_thresholds.values()
        ):
            raise ValueError("calibration thresholds must be between zero and one")
        if len(self.isotonic_x) < 2 or len(self.isotonic_x) != len(self.isotonic_y):
            raise ValueError("calibration isotonic coordinates must have equal usable lengths")
        if any(left >= right for left, right in zip(self.isotonic_x, self.isotonic_x[1:])):
            raise ValueError("calibration isotonic_x must be strictly increasing")
        if any(left > right for left, right in zip(self.isotonic_y, self.isotonic_y[1:])) or any(
            not 0.0 <= value <= 1.0 for value in self.isotonic_y
        ):
            raise ValueError("calibration isotonic_y must be monotonic probabilities")
        if not isinstance(self.heldout_metrics, dict):
            raise TypeError("calibration heldout_metrics must be an object")
        try:
            json.dumps(self.heldout_metrics, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("calibration heldout_metrics must be finite JSON") from error

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        model_revision: str,
        preprocess_version: str,
        corpus_sha256: str,
    ) -> CalibrationArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("OCR calibration artifact must be a JSON object")
        for field, expected in {
            "model_revision": model_revision,
            "preprocess_version": preprocess_version,
            "corpus_sha256": corpus_sha256,
        }.items():
            if payload.get(field) != expected:
                raise ValueError(f"OCR calibration {field} fingerprint mismatch")
        payload["isotonic_x"] = tuple(payload["isotonic_x"])
        payload["isotonic_y"] = tuple(payload["isotonic_y"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "isotonic_x": list(self.isotonic_x),
            "isotonic_y": list(self.isotonic_y),
        }


@dataclass(frozen=True)
class ConfidenceDecision:
    score: float
    confidence_kind: Literal["calibrated", "heuristic"]
    profile: AcceptanceProfile
    threshold: float
    no_text_probability: float | None
    accepted: bool


def profile_for_text(text: str) -> AcceptanceProfile:
    clusters = tuple(cluster for cluster in grapheme_clusters(text) if not cluster.isspace())
    cjk = sum(
        bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", cluster))
        for cluster in clusters
    )
    latin = sum(bool(re.search(r"[A-Za-z]", cluster)) for cluster in clusters)
    if clusters and len(clusters) <= 2 and cjk == len(clusters):
        return AcceptanceProfile.SHORT_CJK
    if clusters and latin and not cjk and len(clusters) <= 12:
        return AcceptanceProfile.LATIN_SFX
    return AcceptanceProfile.DIALOGUE


def linear_score(
    features: OCRConfidenceFeatures,
    coefficients: dict[str, float] = DEFAULT_COEFFICIENTS,
    intercept: float = 0.38,
) -> float:
    normalized = features.normalized()
    return float(
        np.clip(
            intercept + sum(coefficients.get(name, 0.0) * normalized[name] for name in FEATURE_NAMES),
            0,
            1,
        )
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _no_text_probability(
    features: OCRConfidenceFeatures,
    artifact: CalibrationArtifact,
) -> float:
    normalized = features.normalized()
    logit = artifact.no_text_intercept + sum(
        artifact.no_text_coefficients.get(name, 0.0) * normalized[name]
        for name in FEATURE_NAMES
    )
    return _sigmoid(logit)


def decide_ocr_confidence(
    text: str,
    features: OCRConfidenceFeatures,
    artifact: CalibrationArtifact | None = None,
) -> ConfidenceDecision:
    profile = profile_for_text(text)
    if artifact is None:
        score = linear_score(features)
        thresholds = {
            AcceptanceProfile.DIALOGUE: 0.62,
            AcceptanceProfile.SHORT_CJK: 0.48,
            AcceptanceProfile.LATIN_SFX: 0.50,
        }
        threshold = thresholds[profile]
        accepted = bool(text.strip()) and score >= threshold
        return ConfidenceDecision(score, "heuristic", profile, threshold, None, accepted)

    raw = linear_score(features, artifact.coefficients, artifact.intercept)
    score = float(np.interp(raw, artifact.isotonic_x, artifact.isotonic_y))
    no_text_probability = _no_text_probability(features, artifact)
    threshold = artifact.profile_thresholds[profile.value]
    accepted = (
        bool(text.strip())
        and score >= threshold
        and no_text_probability < artifact.no_text_threshold
    )
    return ConfidenceDecision(
        score,
        "calibrated",
        profile,
        threshold,
        no_text_probability,
        accepted,
    )
