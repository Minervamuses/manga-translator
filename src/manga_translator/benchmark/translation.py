"""Human-reviewed Japanese to Taiwan Traditional Chinese benchmark and switch gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

REQUIRED_CATEGORIES = frozenset({"cross_panel", "pronoun", "entity", "tone", "sfx", "caption"})
MQMSeverity = Literal["critical", "major", "minor"]


@dataclass(frozen=True, slots=True)
class TranslationCorpusUnit:
    unit_id: str
    title: str
    source_ja: str
    reference_zh_tw: str
    categories: tuple[str, ...]
    reviewer_id: str
    reviewer_qualified_ja_zh_tw: bool
    disagreement: bool = False
    adjudication: str | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("unit_id", self.unit_id),
            ("title", self.title),
            ("source_ja", self.source_ja),
            ("reference_zh_tw", self.reference_zh_tw),
            ("reviewer_id", self.reviewer_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"translation corpus {field} must not be empty")
        if isinstance(self.categories, (str, bytes)) or not isinstance(
            self.categories, Sequence
        ):
            raise TypeError("translation corpus categories must be a sequence")
        categories = tuple(self.categories)
        if not categories or any(
            not isinstance(category, str) or category not in REQUIRED_CATEGORIES
            for category in categories
        ):
            raise ValueError("translation corpus categories contain unsupported values")
        if len(set(categories)) != len(categories):
            raise ValueError("translation corpus categories must be unique per unit")
        object.__setattr__(self, "categories", categories)
        if not isinstance(self.reviewer_qualified_ja_zh_tw, bool):
            raise TypeError("reviewer qualification must be a boolean")
        if not isinstance(self.disagreement, bool):
            raise TypeError("disagreement must be a boolean")
        if self.adjudication is not None and (
            not isinstance(self.adjudication, str) or not self.adjudication.strip()
        ):
            raise ValueError("adjudication must be non-empty when provided")
        if self.disagreement and self.adjudication is None:
            raise ValueError("review disagreements require adjudication")


@dataclass(frozen=True, slots=True)
class MQMError:
    severity: MQMSeverity
    category: str
    description: str

    def __post_init__(self) -> None:
        if self.severity not in {"critical", "major", "minor"}:
            raise ValueError("unsupported MQM severity")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("MQM category must not be empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("MQM description must not be empty")


@dataclass(frozen=True, slots=True)
class TranslationAssessment:
    unit_id: str
    output_zh_tw: str
    mapping_error: bool = False
    omission: bool = False
    hallucination: bool = False
    mqm_errors: tuple[MQMError, ...] = ()
    approved_names_required: tuple[str, ...] = ()
    approved_names_consistent: bool = True
    taiwan_usage_ok: bool = True
    layout_overflow: bool = False
    repair_attempted: bool = False
    repair_triggered_by_overflow: bool = False
    repair_succeeded: bool = False
    repair_cost: float = 0.0
    visual_triggered: bool = False
    visual_quality_delta: float = 0.0
    visual_cost: float = 0.0
    visual_privacy_profile: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("assessment unit_id must not be empty")
        if not isinstance(self.output_zh_tw, str) or not self.output_zh_tw.strip():
            raise ValueError("assessment output_zh_tw must not be empty")
        boolean_fields = (
            "mapping_error",
            "omission",
            "hallucination",
            "approved_names_consistent",
            "taiwan_usage_ok",
            "layout_overflow",
            "repair_attempted",
            "repair_triggered_by_overflow",
            "repair_succeeded",
            "visual_triggered",
        )
        if any(not isinstance(getattr(self, field), bool) for field in boolean_fields):
            raise TypeError("assessment flags must be booleans")
        if isinstance(self.mqm_errors, (str, bytes)) or not isinstance(
            self.mqm_errors, Sequence
        ):
            raise TypeError("mqm_errors must be a sequence")
        mqm_errors = tuple(self.mqm_errors)
        if any(not isinstance(error, MQMError) for error in mqm_errors):
            raise TypeError("mqm_errors must contain MQMError values")
        object.__setattr__(self, "mqm_errors", mqm_errors)
        if isinstance(self.approved_names_required, (str, bytes)) or not isinstance(
            self.approved_names_required, Sequence
        ):
            raise TypeError("approved names must be a sequence")
        approved_names = tuple(self.approved_names_required)
        if any(
            not isinstance(name, str) or not name.strip()
            for name in approved_names
        ):
            raise ValueError("approved names must be non-empty strings")
        if len(set(approved_names)) != len(approved_names):
            raise ValueError("approved names must be unique")
        object.__setattr__(self, "approved_names_required", approved_names)
        for field in ("repair_cost", "visual_quality_delta", "visual_cost"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{field} must be finite")
        if self.repair_cost < 0 or self.visual_cost < 0:
            raise ValueError("assessment costs must be non-negative")
        if (self.repair_triggered_by_overflow or self.repair_succeeded) and not self.repair_attempted:
            raise ValueError("repair outcome flags require repair_attempted")
        if self.repair_cost and not self.repair_attempted:
            raise ValueError("repair cost requires repair_attempted")
        if self.visual_triggered:
            if not isinstance(self.visual_privacy_profile, str) or not (
                self.visual_privacy_profile.strip()
            ):
                raise ValueError("visual escalation requires a privacy profile")
        elif (
            self.visual_quality_delta != 0
            or self.visual_cost != 0
            or self.visual_privacy_profile is not None
        ):
            raise ValueError("visual metrics require visual_triggered")


def validate_translation_corpus(manifest: dict[str, Any]) -> tuple[TranslationCorpusUnit, ...]:
    if not isinstance(manifest, dict):
        raise TypeError("translation corpus manifest must be an object")
    raw_units = manifest.get("units", [])
    if not isinstance(raw_units, list) or any(not isinstance(item, Mapping) for item in raw_units):
        raise TypeError("translation corpus units must be a list of objects")
    units = tuple(TranslationCorpusUnit(**dict(item)) for item in raw_units)
    if len(units) < 200:
        raise ValueError("translation corpus requires at least 200 human-reviewed units")
    if len({unit.title for unit in units}) < 3:
        raise ValueError("translation corpus requires at least three titles")
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("translation corpus unit IDs must be unique")
    if any(not unit.reviewer_qualified_ja_zh_tw for unit in units):
        raise ValueError("every unit requires qualified Japanese to zh-TW human review")
    present = {category for unit in units for category in unit.categories}
    missing = REQUIRED_CATEGORIES - present
    if missing:
        raise ValueError(f"translation corpus missing categories: {sorted(missing)}")
    counts = manifest.get("counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise TypeError("translation corpus counts must be an object")
        expected_counts = {
            "human_reviewed_units": len(units),
            "titles": len({unit.title for unit in units}),
            "qualified_reviewers": len({unit.reviewer_id for unit in units}),
        }
        if any(counts.get(name) != value for name, value in expected_counts.items()):
            raise ValueError("translation corpus declared counts do not match its units")
    if manifest.get("status") == "blocked":
        raise ValueError("translation corpus status remains blocked")
    return units


def evaluate_translation_assessments(
    units: tuple[TranslationCorpusUnit, ...],
    assessments: tuple[TranslationAssessment, ...],
) -> dict[str, Any]:
    if any(not isinstance(unit, TranslationCorpusUnit) for unit in units):
        raise TypeError("units must contain TranslationCorpusUnit values")
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("translation corpus unit IDs must be unique")
    if any(not isinstance(item, TranslationAssessment) for item in assessments):
        raise TypeError("assessments must contain TranslationAssessment values")
    expected = {unit.unit_id for unit in units}
    assessment_counts = Counter(assessment.unit_id for assessment in assessments)
    duplicate_ids = {unit_id for unit_id, count in assessment_counts.items() if count > 1}
    unknown_ids = set(assessment_counts) - expected
    missing_ids = expected - set(assessment_counts)
    by_id = {assessment.unit_id: assessment for assessment in assessments}
    mapping_complete = not duplicate_ids and not unknown_ids and not missing_ids
    material = [by_id[unit.unit_id] for unit in units if unit.unit_id in by_id]
    severities = Counter(error.severity for item in material for error in item.mqm_errors)
    name_required = [item for item in material if item.approved_names_required]
    repairs = [item for item in material if item.repair_attempted]
    visual = [item for item in material if item.visual_triggered]
    return {
        "mapping_complete": mapping_complete,
        "mapping_errors": sum(item.mapping_error for item in material)
        + len(missing_ids)
        + len(unknown_ids)
        + len(duplicate_ids),
        "missing_ids": sorted(missing_ids),
        "unknown_ids": sorted(unknown_ids),
        "duplicate_ids": sorted(duplicate_ids),
        "omissions": sum(item.omission for item in material),
        "hallucinations": sum(item.hallucination for item in material),
        "mqm": {severity: severities[severity] for severity in ("critical", "major", "minor")},
        "approved_name_consistency": sum(item.approved_names_consistent for item in name_required)
        / max(1, len(name_required)),
        "approved_name_units": len(name_required),
        "taiwan_usage": sum(item.taiwan_usage_ok for item in material) / max(1, len(material)),
        "layout_overflow": sum(item.layout_overflow for item in material),
        "repair": {
            "attempted": len(repairs),
            "outside_overflow_subset": sum(
                item.repair_attempted and not item.repair_triggered_by_overflow for item in material
            ),
            "succeeded": sum(item.repair_succeeded for item in repairs),
            "success_rate": sum(item.repair_succeeded for item in repairs) / max(1, len(repairs)),
            "cost": sum(item.repair_cost for item in repairs),
        },
        "visual": {
            "trigger_rate": len(visual) / max(1, len(material)),
            "mean_quality_delta": sum(item.visual_quality_delta for item in visual)
            / max(1, len(visual)),
            "cost": sum(item.visual_cost for item in visual),
            "privacy_profiles": sorted(
                {item.visual_privacy_profile for item in visual if item.visual_privacy_profile}
            ),
        },
    }


def evaluate_translation_switch_gate(
    units: tuple[TranslationCorpusUnit, ...],
    baseline_assessments: tuple[TranslationAssessment, ...],
    candidate_assessments: tuple[TranslationAssessment, ...],
) -> dict[str, Any]:
    baseline = evaluate_translation_assessments(units, baseline_assessments)
    candidate = evaluate_translation_assessments(units, candidate_assessments)
    baseline_by_id = {item.unit_id: item for item in baseline_assessments}
    candidate_by_id = {item.unit_id: item for item in candidate_assessments}
    repairs = candidate["repair"]
    repair_ids = {
        item.unit_id for item in candidate_assessments if item.repair_attempted
    }
    compact_positive = repairs["attempted"] == 0 or (
        repairs["outside_overflow_subset"] == 0
        and repairs["succeeded"] > 0
        and candidate["layout_overflow"] < baseline["layout_overflow"]
        and all(
            unit_id in baseline_by_id
            and baseline_by_id[unit_id].layout_overflow
            and candidate_by_id[unit_id].repair_succeeded
            and not candidate_by_id[unit_id].layout_overflow
            for unit_id in repair_ids
        )
    )
    approved_name_scope_unchanged = baseline["mapping_complete"] and candidate[
        "mapping_complete"
    ] and all(
        baseline_by_id[unit.unit_id].approved_names_required
        == candidate_by_id[unit.unit_id].approved_names_required
        for unit in units
    )
    checks = {
        "baseline_mapping_complete": baseline["mapping_complete"],
        "mapping_error_zero": candidate["mapping_complete"] and candidate["mapping_errors"] == 0,
        "approved_name_scope_unchanged": approved_name_scope_unchanged,
        "approved_name_consistency_100_percent": candidate["approved_name_consistency"] == 1.0,
        "critical_not_worse": candidate["mqm"]["critical"] <= baseline["mqm"]["critical"],
        "major_not_worse": candidate["mqm"]["major"] <= baseline["mqm"]["major"],
        "omissions_not_worse": candidate["omissions"] <= baseline["omissions"],
        "hallucinations_not_worse": candidate["hallucinations"]
        <= baseline["hallucinations"],
        "taiwan_usage_not_worse": candidate["taiwan_usage"] >= baseline["taiwan_usage"],
        "compact_repair_positive_on_overflow_only": compact_positive,
    }
    baseline_weighted = (
        baseline["mqm"]["critical"] * 10
        + baseline["mqm"]["major"] * 3
        + baseline["mqm"]["minor"]
        + baseline["omissions"] * 3
        + baseline["hallucinations"] * 3
    )
    candidate_weighted = (
        candidate["mqm"]["critical"] * 10
        + candidate["mqm"]["major"] * 3
        + candidate["mqm"]["minor"]
        + candidate["omissions"] * 3
        + candidate["hallucinations"] * 3
    )
    passed = all(checks.values())
    substantive_improvement = candidate_weighted < baseline_weighted
    return {
        "status": "passed" if passed else "blocked",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "substantive_quality_improvement": substantive_improvement,
        "model_upgrade_claim": "allowed" if passed and substantive_improvement else "not_allowed",
        "contract_reliability_claim": "allowed" if passed else "not_allowed",
        "translation_flow_switch": "allowed" if passed else "not_performed",
    }
