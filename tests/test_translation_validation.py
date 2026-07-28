from __future__ import annotations

import asyncio

from manga_translator.translation.repair import RepairCoordinator, RepairUnit
from manga_translator.translation.validate import (
    TranslationInput,
    TranslationIssue,
    normalize_display_text,
    validate_translation_batch,
)
from manga_translator.typography.layout import LayoutOverflow


def _run(coro):
    return asyncio.run(coro)


def _issue(code: str, unit_id: str = "u0001") -> TranslationIssue:
    return TranslationIssue(code, unit_id, code)


def _overflow() -> LayoutOverflow:
    return LayoutOverflow(
        available_size=(80, 40),
        grapheme_count=12,
        suggested_max_graphemes=7,
        suggested_max_lines=2,
        reason="no hard-valid candidate",
        rejected=(("outside_safe_region", 3),),
    )


def test_quote_dash_ellipsis_and_fullwidth_punctuation_golden_are_lossless() -> None:
    golden = (
        "「你真的要走……？」",
        "『不——我不走。』",
        "他說：「等等！」",
        "Ａ：這是全形標點；Ｂ：保留。",
        "……只差一步——",
    )
    assert tuple(normalize_display_text(text) for text in golden) == golden


def test_validator_reports_deterministic_issues_without_mutating_raw_response() -> None:
    raw = "「米卡不會來——」\n\u200b"
    inputs = (
        TranslationInput("u0001", "ミカは来ない", raw, entity_refs=("米卡",)),
        TranslationInput("u0002", "どうしたの", "どうしたの"),
        TranslationInput("u0003", "猫だ", "同一句"),
        TranslationInput("u0004", "犬だ", "同一句"),
        TranslationInput("u0005", "短い", "重複內容重複內容"),
        TranslationInput("unknown", "はい", "這是一段過度冗長的譯文" * 8),
    )

    result = validate_translation_batch(
        inputs,
        expected_ids=("u0001", "u0002", "u0003", "u0004", "u0005", "u0006"),
        approved_entities={"ミカ": "米卡"},
    )
    codes = {issue.code for issue in result.issues}

    assert inputs[0].raw_translation == raw
    assert inputs[0].display_translation == "「米卡不會來——」"
    assert {
        "source_echo",
        "untranslated_japanese_ratio",
        "repeated_content",
        "cross_unit_duplication",
        "length_anomaly",
        "illegal_id",
    } <= codes


def test_missing_entity_and_approved_glossary_mismatch_are_separate_issues() -> None:
    result = validate_translation_batch(
        (TranslationInput("u0001", "ミカが来た", "她來了", entity_refs=("米卡",)),),
        approved_entities={"ミカ": "米卡"},
    )
    assert {issue.code for issue in result.issues} == {
        "missing_entity",
        "approved_glossary_mismatch",
    }


def test_targeted_repair_only_sends_problem_units_and_keeps_exact_ids() -> None:
    calls = []

    async def provider(items):
        calls.append(items)
        return [{"id": item.id, "translation": "米卡會來"} for item in reversed(items)]

    bad = RepairUnit(
        "u0001",
        "ミカが来る",
        "過度冗長的舊譯",
        issues=(_issue("length_anomaly"),),
        must_preserve_entities=("米卡",),
    )
    good = RepairUnit("u0002", "犬だ", "是狗")
    outcomes = _run(RepairCoordinator(provider).targeted_repair((bad, good)))

    assert [[item.id for item in call] for call in calls] == [["u0001"]]
    assert outcomes["u0001"].final_text == "米卡會來"
    assert outcomes["u0001"].revision.original_text == "過度冗長的舊譯"  # type: ignore[union-attr]
    assert outcomes["u0001"].revision.reason_codes == ("length_anomaly",)  # type: ignore[union-attr]
    assert outcomes["u0002"].kept_original


def test_targeted_repair_contract_failure_is_no_erase_and_bounded() -> None:
    calls = 0

    async def provider(_items):
        nonlocal calls
        calls += 1
        return [{"id": "wrong", "translation": "錯誤"}]

    unit = RepairUnit("u0001", "猫だ", "原譯", issues=(_issue("length_anomaly"),))
    coordinator = RepairCoordinator(provider, max_revisions_per_unit=1)
    first = _run(coordinator.targeted_repair((unit,)))["u0001"]
    second = _run(coordinator.targeted_repair((unit,)))["u0001"]

    assert calls == 1
    assert first.kept_original and first.final_text == "原譯"
    assert second.kept_original and second.final_text == "原譯"


def test_compact_repair_cannot_drop_entity_or_critical_negation() -> None:
    calls = 0

    async def provider(items):
        nonlocal calls
        calls += 1
        assert items[0].max_graphemes == 7
        assert items[0].max_lines == 2
        return [{"id": items[0].id, "translation": "快走"}]

    unit = RepairUnit(
        "u0001",
        "ミカは行かない",
        "米卡不會走",
        must_preserve_entities=("米卡",),
        must_preserve_facts=("走",),
    )
    outcome = _run(
        RepairCoordinator(provider).compact_repair(
            unit,
            _overflow(),
            layout_check=lambda _text: object(),  # type: ignore[arg-type,return-value]
        )
    )

    assert calls == 1
    assert outcome.kept_original
    assert outcome.final_text == "米卡不會走"
    assert {issue.code for issue in outcome.issues} >= {
        "repair_dropped_entity",
        "repair_dropped_negation",
    }


def test_compact_repair_only_runs_on_overflow_and_must_pass_layout_again() -> None:
    calls = 0

    async def provider(items):
        nonlocal calls
        calls += 1
        return [{"id": items[0].id, "translation": "米卡不走"}]

    unit = RepairUnit(
        "u0001", "ミカは行かない", "米卡絕對不會離開這個地方", must_preserve_entities=("米卡",)
    )
    coordinator = RepairCoordinator(provider)
    ordinary = _run(
        coordinator.compact_repair(
            unit,
            object(),
            layout_check=lambda _text: object(),  # type: ignore[arg-type,return-value]
        )
    )
    still_overflows = _run(
        coordinator.compact_repair(unit, _overflow(), layout_check=lambda _text: _overflow())
    )
    accepted = _run(
        coordinator.compact_repair(
            unit,
            _overflow(),
            layout_check=lambda _text: object(),  # type: ignore[arg-type,return-value]
        )
    )

    assert ordinary.kept_original and calls == 2
    assert still_overflows.kept_original
    assert accepted.final_text == "米卡不走"
    assert not accepted.kept_original
