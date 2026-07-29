from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from manga_translator.benchmark.cli import main as benchmark_main
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

    unknown_category = _manifest()
    unknown_category["units"][0]["categories"] = ["unsupported"]
    with pytest.raises(ValueError, match="unsupported"):
        validate_translation_corpus(unknown_category)

    stale_counts = _manifest()
    stale_counts["counts"] = {
        "human_reviewed_units": 0,
        "titles": 0,
        "qualified_reviewers": 0,
    }
    with pytest.raises(ValueError, match="declared counts"):
        validate_translation_corpus(stale_counts)

    blocked = _manifest()
    blocked["status"] = "blocked"
    with pytest.raises(ValueError, match="remains blocked"):
        validate_translation_corpus(blocked)


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

    unrelated_compact = list(candidate)
    unrelated_compact[1] = replace(
        unrelated_compact[1],
        repair_attempted=False,
        repair_triggered_by_overflow=False,
        repair_succeeded=False,
        repair_cost=0.0,
    )
    unrelated_compact[2] = replace(
        unrelated_compact[2],
        repair_attempted=True,
        repair_triggered_by_overflow=True,
        repair_succeeded=True,
    )
    blocked_unpaired = evaluate_translation_switch_gate(
        units, baseline, tuple(unrelated_compact)
    )
    assert not blocked_unpaired["checks"]["compact_repair_positive_on_overflow_only"]


def test_switch_gate_requires_complete_baseline_and_stable_name_scope() -> None:
    units = validate_translation_corpus(_manifest())
    baseline = _assessments(units, baseline=True)
    candidate = list(_assessments(units))

    incomplete = evaluate_translation_switch_gate(units, baseline[:-1], tuple(candidate))
    assert incomplete["status"] == "blocked"
    assert not incomplete["checks"]["baseline_mapping_complete"]

    candidate[2] = replace(candidate[2], approved_names_required=("另一個名字",))
    changed_scope = evaluate_translation_switch_gate(units, baseline, tuple(candidate))
    assert changed_scope["status"] == "blocked"
    assert not changed_scope["checks"]["approved_name_scope_unchanged"]


def test_assessment_rejects_non_finite_or_internally_inconsistent_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        TranslationAssessment("u1", "譯文", repair_cost=math.nan)
    with pytest.raises(ValueError, match="repair outcome"):
        TranslationAssessment("u1", "譯文", repair_succeeded=True)
    with pytest.raises(ValueError, match="privacy profile"):
        TranslationAssessment("u1", "譯文", visual_triggered=True)
    with pytest.raises(TypeError, match="approved names"):
        TranslationAssessment("u1", "譯文", approved_names_required="米卡")  # type: ignore[arg-type]


def test_switch_gate_blocks_new_omissions_or_hallucinations() -> None:
    units = validate_translation_corpus(_manifest())
    baseline = _assessments(units, baseline=True)
    candidate = list(_assessments(units))
    candidate[0] = replace(candidate[0], omission=True, hallucination=True)

    result = evaluate_translation_switch_gate(units, baseline, tuple(candidate))
    assert result["status"] == "blocked"
    assert not result["checks"]["omissions_not_worse"]
    assert not result["checks"]["hallucinations_not_worse"]


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


def test_translation_validate_cli_reports_invalid_json_as_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text("{", encoding="utf-8")

    exit_code = benchmark_main(
        ["--root", str(tmp_path), "translation-validate", "--manifest", str(manifest)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "blocked"
