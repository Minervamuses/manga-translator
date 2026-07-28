"""Human-reviewed Japanese to Taiwan Traditional Chinese benchmark and switch gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class MQMError:
    severity: MQMSeverity
    category: str
    description: str


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


def validate_translation_corpus(manifest: dict[str, Any]) -> tuple[TranslationCorpusUnit, ...]:
    units = tuple(TranslationCorpusUnit(**item) for item in manifest.get("units", []))
    if len(units) < 200:
        raise ValueError("translation corpus requires at least 200 human-reviewed units")
    if len({unit.title for unit in units}) < 3:
        raise ValueError("translation corpus requires at least three titles")
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("translation corpus unit IDs must be unique")
    if any(
        not unit.source_ja.strip()
        or not unit.reference_zh_tw.strip()
        or not unit.reviewer_id.strip()
        or not unit.reviewer_qualified_ja_zh_tw
        for unit in units
    ):
        raise ValueError("every unit requires qualified Japanese to zh-TW human review")
    if any(unit.disagreement and not (unit.adjudication or "").strip() for unit in units):
        raise ValueError("review disagreements require adjudication")
    present = {category for unit in units for category in unit.categories}
    missing = REQUIRED_CATEGORIES - present
    if missing:
        raise ValueError(f"translation corpus missing categories: {sorted(missing)}")
    return units


def evaluate_translation_assessments(
    units: tuple[TranslationCorpusUnit, ...],
    assessments: tuple[TranslationAssessment, ...],
) -> dict[str, Any]:
    expected = {unit.unit_id for unit in units}
    by_id = {assessment.unit_id: assessment for assessment in assessments}
    mapping_complete = len(by_id) == len(assessments) == len(units) and set(by_id) == expected
    material = [by_id[unit.unit_id] for unit in units if unit.unit_id in by_id]
    severities = Counter(error.severity for item in material for error in item.mqm_errors)
    name_required = [item for item in material if item.approved_names_required]
    repairs = [item for item in material if item.repair_attempted]
    visual = [item for item in material if item.visual_triggered]
    return {
        "mapping_complete": mapping_complete,
        "mapping_errors": sum(item.mapping_error for item in material)
        + (len(units) - len(material)),
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
    repairs = candidate["repair"]
    compact_positive = repairs["attempted"] == 0 or (
        repairs["outside_overflow_subset"] == 0
        and repairs["succeeded"] > 0
        and candidate["layout_overflow"] < baseline["layout_overflow"]
    )
    checks = {
        "mapping_error_zero": candidate["mapping_complete"] and candidate["mapping_errors"] == 0,
        "approved_name_consistency_100_percent": candidate["approved_name_consistency"] == 1.0,
        "critical_not_worse": candidate["mqm"]["critical"] <= baseline["mqm"]["critical"],
        "major_not_worse": candidate["mqm"]["major"] <= baseline["mqm"]["major"],
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
