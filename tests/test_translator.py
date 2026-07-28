from __future__ import annotations

from manga_translator.config import OpenRouterConfig
from manga_translator.translator import (
    _build_prompt_with_context,
    _parse_response,
    sanitize_translation_text,
    validate_translation,
)


def cfg(**updates) -> OpenRouterConfig:
    base = OpenRouterConfig(api_key="test", model="test/model")
    return base.model_copy(update=updates)


def test_parse_json_response_by_stable_ids() -> None:
    response = '{"translations":[{"id":"T0000","text":"你好"},{"id":"T0001","text":"走吧"}]}'
    assert _parse_response(response, 2) == ["你好", "走吧"]


def test_single_item_repair_accepts_dict_even_when_model_reuses_original_id() -> None:
    response = '{"translations":[{"id":"T0017","text":"別擔心"}]}'
    assert _parse_response(response, 1) == ["別擔心"]


def test_parse_numbered_lines_without_copying_labels() -> None:
    response = "[0]：你好\n[1]：走吧"
    assert _parse_response(response, 2) == ["你好", "走吧"]


def test_sanitizer_repairs_common_mojibake_and_duplicate_lines() -> None:
    assert sanitize_translation_text("翻譯：等等â€¦\n翻譯：等等â€¦") == "等等…"


def test_sanitizer_removes_model_separators_and_source_absent_ellipsis() -> None:
    assert sanitize_translation_text("||你好丨", source="こんにちは") == "你好"
    assert sanitize_translation_text("你好...", source="こんにちは") == "你好"
    assert sanitize_translation_text("等等…", source="待って…") == "等等…"
    assert sanitize_translation_text("||...", source="こんにちは") == ""


def test_validation_accepts_normal_traditional_chinese() -> None:
    assert validate_translation("どうしたの？", "你怎麼了？", cfg()).valid


def test_validation_rejects_empty_copied_japanese_and_garble() -> None:
    assert not validate_translation("どうしたの？", "", cfg()).valid
    assert "source_copied" in validate_translation("どうしたの？", "どうしたの？", cfg()).issues
    assert "mojibake" in validate_translation("待って", "ç­‰ä¸€ä¸‹Ã", cfg()).issues
    assert "empty" in validate_translation("待って", "||...", cfg()).issues


def test_validation_rejects_implausibly_long_output() -> None:
    result = validate_translation("はい", "這是一段完全不合理而且長得遠超出原文的模型解說文字" * 3, cfg())
    assert "too_long" in result.issues


def test_context_prompt_only_requests_the_target_id() -> None:
    prompt = _build_prompt_with_context(["前文", "目標", "後文"], index=1, context_size=1)
    assert "只輸出 role=target 的 T0001" in prompt
    assert "每個輸入 id 必須剛好輸出一次" not in prompt


def test_sanitizer_removes_long_sound_mark_mistranslated_as_dashes() -> None:
    assert (
        sanitize_translation_text(
            "謝謝指導——！",
            source="ありがとうございましたーッ",
        )
        == "謝謝指導!"
    )
    assert sanitize_translation_text("謝謝指導︱！", source="ありがとうございましたーッ") == "謝謝指導!"


def test_sanitizer_preserves_real_semantic_dash_from_source() -> None:
    assert sanitize_translation_text("等等——別過來！", source="待て――来るな！") == "等等——別過來!"


def test_sanitizer_collapses_accidental_long_adjacent_duplicate() -> None:
    assert (
        sanitize_translation_text(
            "情報已經掌握了情報已經掌握了",
            source="情報はもう掴んだ",
        )
        == "情報已經掌握了"
    )


def test_sanitizer_keeps_short_expressive_or_source_repetition() -> None:
    assert sanitize_translation_text("不要不要", source="やめて") == "不要不要"
    assert sanitize_translation_text("快跑快跑快跑快跑", source="逃げろ逃げろ") == "快跑快跑快跑快跑"
