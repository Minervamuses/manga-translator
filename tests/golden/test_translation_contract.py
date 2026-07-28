from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from manga_translator.benchmark.translation import (
    MQMError,
    TranslationAssessment,
    evaluate_translation_assessments,
    evaluate_translation_switch_gate,
    validate_translation_corpus,
)


def _manifest(count: int = 200) -> dict:
    categories = ("cross_panel", "pronoun", "entity", "tone", "sfx", "caption")
    return {
        "units": [
            {
                "unit_id": f"u{index:04d}",
                "title": f"title-{index % 3}",
                "source_ja": f"原文{index}",
                "reference_zh_tw": f"台灣譯文{index}",
                "categories": [categories[index % len(categories)]],
                "reviewer_id": f"reviewer-{index % 2}",
                "reviewer_qualified_ja_zh_tw": True,
                "disagreement": index == 0,
                "adjudication": "reviewer consensus" if index == 0 else None,
            }
            for index in range(count)
        ]
    }


def _assessments(units, *, baseline: bool = False):
    material = []
    for index, unit in enumerate(units):
        errors = (
            (MQMError("major", "accuracy", "baseline error"),) if baseline and index == 0 else ()
        )
        material.append(
            TranslationAssessment(
                unit_id=unit.unit_id,
                output_zh_tw=unit.reference_zh_tw,
                mqm_errors=errors,
                approved_names_required=("米卡",) if index == 2 else (),
                approved_names_consistent=True,
                layout_overflow=baseline and index == 1,
                repair_attempted=not baseline and index == 1,
                repair_triggered_by_overflow=not baseline and index == 1,
                repair_succeeded=not baseline and index == 1,
                repair_cost=0.002 if not baseline and index == 1 else 0.0,
                visual_triggered=not baseline and index == 3,
                visual_quality_delta=0.5 if not baseline and index == 3 else 0.0,
                visual_cost=0.004 if not baseline and index == 3 else 0.0,
                visual_privacy_profile="zdr" if not baseline and index == 3 else None,
            )
        )
    return tuple(material)


def test_corpus_requires_200_units_three_titles_categories_and_qualified_review() -> None:
    units = validate_translation_corpus(_manifest())
    assert len(units) == 200
    assert len({unit.title for unit in units}) == 3

    with pytest.raises(ValueError, match="at least 200"):
        validate_translation_corpus(_manifest(199))

    unqualified = _manifest()
    unqualified["units"][0]["reviewer_qualified_ja_zh_tw"] = False
    with pytest.raises(ValueError, match="qualified"):
        validate_translation_corpus(unqualified)

    unadjudicated = _manifest()
    unadjudicated["units"][0]["adjudication"] = None
    with pytest.raises(ValueError, match="adjudication"):
        validate_translation_corpus(unadjudicated)


def test_metrics_cover_mapping_mqm_names_taiwan_layout_repair_and_visual() -> None:
    units = validate_translation_corpus(_manifest())
    metrics = evaluate_translation_assessments(units, _assessments(units))

    assert metrics["mapping_complete"]
    assert metrics["mapping_errors"] == 0
    assert metrics["mqm"] == {"critical": 0, "major": 0, "minor": 0}
    assert metrics["approved_name_consistency"] == 1
    assert metrics["taiwan_usage"] == 1
    assert metrics["repair"] == {
        "attempted": 1,
        "outside_overflow_subset": 0,
        "succeeded": 1,
        "success_rate": 1,
        "cost": 0.002,
    }
    assert metrics["visual"]["trigger_rate"] == 1 / 200
    assert metrics["visual"]["mean_quality_delta"] == 0.5
    assert metrics["visual"]["privacy_profiles"] == ["zdr"]


def test_switch_gate_requires_zero_mapping_100_percent_names_and_safe_compaction() -> None:
    units = validate_translation_corpus(_manifest())
    baseline = _assessments(units, baseline=True)
    candidate = _assessments(units)
    passed = evaluate_translation_switch_gate(units, baseline, candidate)

    assert passed["status"] == "passed"
    assert passed["translation_flow_switch"] == "allowed"
    assert passed["model_upgrade_claim"] == "allowed"

    mapping_error = list(candidate)
    mapping_error[0] = replace(mapping_error[0], mapping_error=True)
    blocked_mapping = evaluate_translation_switch_gate(units, baseline, tuple(mapping_error))
    assert blocked_mapping["status"] == "blocked"
    assert not blocked_mapping["checks"]["mapping_error_zero"]

    invalid_compact = list(candidate)
    invalid_compact[1] = replace(invalid_compact[1], repair_triggered_by_overflow=False)
    blocked_compact = evaluate_translation_switch_gate(units, baseline, tuple(invalid_compact))
    assert not blocked_compact["checks"]["compact_repair_positive_on_overflow_only"]


def test_no_quality_gain_never_claims_model_upgrade() -> None:
    units = validate_translation_corpus(_manifest())
    same = tuple(
        replace(
            item,
            repair_attempted=False,
            repair_triggered_by_overflow=False,
            repair_succeeded=False,
            repair_cost=0.0,
        )
        for item in _assessments(units)
    )
    result = evaluate_translation_switch_gate(units, same, same)

    assert result["status"] == "passed"
    assert not result["substantive_quality_improvement"]
    assert result["model_upgrade_claim"] == "not_allowed"
    assert result["contract_reliability_claim"] == "allowed"


def test_repository_manifest_truthfully_blocks_translation_flow_switch() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "benchmarks/translation_zh_tw_v1/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked"
    assert manifest["counts"]["human_reviewed_units"] == 0
    assert manifest["counts"]["titles"] == 0
    assert manifest["model_upgrade_claim"] == "not_allowed"
    assert manifest["translation_flow_switch"] == "not_performed"
    assert manifest["legacy_context_window_parser"] == "retained_until_gate_passes"
