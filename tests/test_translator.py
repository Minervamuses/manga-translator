from __future__ import annotations

import hashlib
import json

import httpx
import pytest

import manga_translator.translator as translator_module
from manga_translator.config import OpenRouterConfig
from manga_translator.contracts.mapping import MappingContractError, source_sha256
from manga_translator.translator import (
    _build_prompt_with_context,
    _parse_response,
    sanitize_translation_text,
    translate_batch_mapped,
    validate_translation,
)


def cfg(**updates) -> OpenRouterConfig:
    base = OpenRouterConfig(api_key="test", model="test/model")
    return base.model_copy(update=updates)


def _provider_response_bytes(items: list[dict[str, str]]) -> bytes:
    content = json.dumps(
        {"translations": items},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return json.dumps(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _install_response_transport(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[bytes],
) -> None:
    pending = list(responses)
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test"
        return httpx.Response(
            200,
            content=pending.pop(0),
            headers={"content-type": "application/json; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(translator_module.httpx, "AsyncClient", client_factory)


def test_parse_json_response_by_stable_ids() -> None:
    sources = ["こんにちは", "行こう"]
    response = json.dumps(
        {
            "translations": [
                {"id": "T0000", "source_sha256": source_sha256(sources[0]), "text": "你好"},
                {"id": "T0001", "source_sha256": source_sha256(sources[1]), "text": "走吧"},
            ]
        },
        ensure_ascii=False,
    )
    assert _parse_response(
        response, 2, source_hashes=[source_sha256(source) for source in sources]
    ) == ["你好", "走吧"]


def test_mapped_batch_preserves_provider_order_and_exact_response_artifact(
    tmp_path, monkeypatch
) -> None:
    sources = ["こんにちは", "さようなら"]
    item_ids = ["R-test:T0000", "R-test:T0001"]
    raw = _provider_response_bytes(
        [
            {
                "id": item_ids[1],
                "source_sha256": source_sha256(sources[1]),
                "translation": "再見",
            },
            {
                "id": item_ids[0],
                "source_sha256": source_sha256(sources[0]),
                "translation": "你好",
            },
        ]
    )
    _install_response_transport(monkeypatch, [raw])

    responses = translate_batch_mapped(
        sources,
        cfg(validate_translation=False),
        item_ids=item_ids,
        artifact_root=tmp_path,
    )

    assert [response.item_id for response in responses] == item_ids
    assert [response.translation for response in responses] == ["你好", "再見"]
    assert [response.response_index for response in responses] == [1, 0]
    references = [response.raw_response_ref for response in responses]
    assert all(reference is not None for reference in references)
    reference = references[0]
    assert reference == references[1]
    assert reference is not None
    assert reference.sha256 == hashlib.sha256(raw).hexdigest()
    assert reference.size_bytes == len(raw)
    assert reference.media_type == "application/json"
    assert reference.relative_path == (
        f"artifacts/translation-responses/{reference.sha256}.json"
    )
    assert (tmp_path / reference.relative_path).read_bytes() == raw
    assert b"Bearer test" not in raw


def test_mapped_multibatch_response_indexes_are_local_to_each_artifact(
    tmp_path, monkeypatch
) -> None:
    sources = ["一つ", "二つ"]
    item_ids = ["R-multi:T0000", "R-multi:T0001"]
    raw_responses = [
        _provider_response_bytes(
            [
                {
                    "id": item_id,
                    "source_sha256": source_sha256(source),
                    "translation": translation,
                }
            ]
        )
        for source, item_id, translation in zip(
            sources,
            item_ids,
            ["一個", "兩個"],
            strict=True,
        )
    ]
    _install_response_transport(monkeypatch, raw_responses)

    responses = translate_batch_mapped(
        sources,
        cfg(batch_size=1, validate_translation=False),
        item_ids=item_ids,
        artifact_root=tmp_path,
    )

    assert [response.response_index for response in responses] == [0, 0]
    references = [response.raw_response_ref for response in responses]
    assert all(reference is not None for reference in references)
    assert references[0] != references[1]
    for reference, raw in zip(references, raw_responses, strict=True):
        assert reference is not None
        assert reference.relative_path is not None
        assert (tmp_path / reference.relative_path).read_bytes() == raw


def test_repaired_item_references_the_repair_response(tmp_path, monkeypatch) -> None:
    source = "こんにちは"
    item_id = "R-repair:T0000"
    initial_raw = _provider_response_bytes(
        [
            {
                "id": item_id,
                "source_sha256": source_sha256(source),
                "translation": source,
            }
        ]
    )
    repair_raw = _provider_response_bytes(
        [
            {
                "id": item_id,
                "source_sha256": source_sha256(source),
                "translation": "你好",
            }
        ]
    )
    _install_response_transport(monkeypatch, [initial_raw, repair_raw])

    response = translate_batch_mapped(
        [source],
        cfg(content_retries=0),
        item_ids=[item_id],
        artifact_root=tmp_path,
    )[0]

    assert response.translation == "你好"
    assert response.raw_response_ref is not None
    assert response.raw_response_ref.sha256 == hashlib.sha256(repair_raw).hexdigest()
    assert response.raw_response_ref.relative_path is not None
    assert (tmp_path / response.raw_response_ref.relative_path).read_bytes() == repair_raw
    initial_hash = hashlib.sha256(initial_raw).hexdigest()
    assert (
        tmp_path / f"artifacts/translation-responses/{initial_hash}.json"
    ).read_bytes() == initial_raw


def test_malformed_success_response_is_persisted_before_json_decode(
    tmp_path, monkeypatch
) -> None:
    raw = b"not-json"
    _install_response_transport(monkeypatch, [raw])

    with pytest.raises(ValueError):
        translate_batch_mapped(
            ["こんにちは"],
            cfg(validate_translation=False),
            item_ids=["R-malformed:T0000"],
            artifact_root=tmp_path,
        )

    raw_hash = hashlib.sha256(raw).hexdigest()
    artifact = tmp_path / f"artifacts/translation-responses/{raw_hash}.json"
    assert artifact.read_bytes() == raw


def test_exact_id_failure_exposes_the_persisted_batch_artifact(
    tmp_path, monkeypatch
) -> None:
    sources = ["一つ", "二つ"]
    item_ids = ["R-invalid:T0000", "R-invalid:T0001"]
    raw = _provider_response_bytes(
        [
            {
                "id": item_ids[0],
                "source_sha256": source_sha256(sources[0]),
                "translation": "一個",
            },
            {
                "id": item_ids[0],
                "source_sha256": source_sha256(sources[0]),
                "translation": "覆寫",
            },
        ]
    )
    _install_response_transport(monkeypatch, [raw])

    with pytest.raises(MappingContractError) as captured:
        translate_batch_mapped(
            sources,
            cfg(validate_translation=False),
            item_ids=item_ids,
            artifact_root=tmp_path,
        )

    assert len(captured.value.raw_response_refs) == 1
    reference = captured.value.raw_response_refs[0]
    assert reference.sha256 == hashlib.sha256(raw).hexdigest()
    assert reference.relative_path is not None
    assert (tmp_path / reference.relative_path).read_bytes() == raw


def test_single_item_repair_accepts_dict_even_when_model_reuses_original_id() -> None:
    source_hash = source_sha256("心配するな")
    response = json.dumps(
        {
            "translations": [
                {"id": "T0017", "source_sha256": source_hash, "text": "別擔心"}
            ]
        },
        ensure_ascii=False,
    )
    assert _parse_response(
        response, 1, expected_ids=["T0017"], source_hashes=[source_hash]
    ) == ["別擔心"]


def test_parse_numbered_lines_without_copying_labels() -> None:
    response = "[0]：你好\n[1]：走吧"
    with pytest.raises(MappingContractError):
        _parse_response(
            response,
            2,
            source_hashes=[source_sha256("こんにちは"), source_sha256("行こう")],
        )


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
