from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from manga_translator.contracts.mapping import RequestMap, build_request_map
from manga_translator.translation.client import (
    RetryPolicy,
    StructuredOutputUnsupported,
    StructuredTranslationClient,
    TranslationContentError,
    TranslationSchemaError,
)
from manga_translator.translation.provider import ProviderPolicy
from manga_translator.translation.schema import translation_json_schema


def _request(page: str = "page") -> RequestMap:
    return build_request_map(page, [("region-a", "猫だ"), ("region-b", "犬だ")])


def _items(request: RequestMap) -> list[dict[str, str]]:
    return [
        {"id": item.item_id, "translation": translation}
        for item, translation in zip(request.items, ("是貓", "是狗"), strict=True)
    ]


def _provider_response(
    items: list[dict[str, object]],
    *,
    provider: str = "fixture-provider",
    model: str = "fixture/model",
) -> httpx.Response:
    content = json.dumps({"translations": items}, ensure_ascii=False)
    return httpx.Response(
        200,
        json={
            "provider": provider,
            "model": model,
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            "choices": [{"message": {"content": content}}],
        },
    )


def _run(coro):
    return asyncio.run(coro)


def test_schema_is_strict_and_payload_denies_collection_by_default() -> None:
    request = _request()
    captured: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(http_request.content))
        return _provider_response(_items(request))

    async def scenario():
        async with StructuredTranslationClient(
            api_key="top-secret",
            model="requested/model",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.translate(request, "translate this page")

    result = _run(scenario())
    payload = captured[0]
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"  # type: ignore[index]
    assert response_format["json_schema"]["strict"] is True  # type: ignore[index]
    assert payload["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    schema = translation_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["StructuredTranslationItem"]["additionalProperties"] is False
    assert "top-secret" not in json.dumps(result.provenance.sanitized_request)
    assert result.provenance.actual_provider == "fixture-provider"
    assert result.provenance.actual_model == "fixture/model"
    assert result.provenance.usage["completion_tokens"] == 6
    assert result.provenance.raw_response
    assert len(result.provenance.schema_hash) == 64
    assert len(result.provenance.prompt_hash) == 64


def test_reordered_array_still_maps_by_exact_id() -> None:
    request = _request()
    transport = httpx.MockTransport(lambda _request: _provider_response(list(reversed(_items(request)))))

    async def scenario():
        async with StructuredTranslationClient(
            api_key="test", model="test/model", transport=transport
        ) as client:
            return await client.translate(request, "prompt")

    result = _run(scenario())
    assert result.batch.by_region_key == {"region-a": "是貓", "region-b": "是狗"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda items: items[:1],
        lambda items: [items[0], items[0]],
        lambda items: [*items, {"id": "extra", "translation": "額外"}],
        lambda items: [items[0], {**items[1], "id": "unknown"}],
    ],
    ids=["missing", "duplicate", "extra", "unknown"],
)
def test_exact_id_failures_reject_the_whole_response(
    mutation: Callable[[list[dict[str, str]]], list[dict[str, str]]],
) -> None:
    request = _request()
    transport = httpx.MockTransport(lambda _request: _provider_response(mutation(_items(request))))

    async def scenario():
        async with StructuredTranslationClient(
            api_key="test",
            model="test/model",
            transport=transport,
            retry_policy=RetryPolicy(content=0),
        ) as client:
            await client.translate(request, "prompt")

    with pytest.raises(TranslationContentError):
        _run(scenario())


def test_schema_mismatch_and_unsupported_provider_fail_safely() -> None:
    request = _request()
    bad_schema = [{**_items(request)[0], "unexpected": True}, _items(request)[1]]

    async def schema_scenario():
        async with StructuredTranslationClient(
            api_key="test",
            model="test/model",
            transport=httpx.MockTransport(lambda _request: _provider_response(bad_schema)),
            retry_policy=RetryPolicy(schema=0),
        ) as client:
            await client.translate(request, "prompt")

    with pytest.raises(TranslationSchemaError):
        _run(schema_scenario())

    async def unsupported_scenario():
        response = httpx.Response(400, text="response_format json_schema is unsupported")
        async with StructuredTranslationClient(
            api_key="test",
            model="test/model",
            transport=httpx.MockTransport(lambda _request: response),
        ) as client:
            await client.translate(request, "prompt")

    with pytest.raises(StructuredOutputUnsupported):
        _run(unsupported_scenario())


def test_one_client_reuses_job_pool_across_pages_and_closes_explicitly() -> None:
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(http_request.content)
        ids = [entry["id"] for entry in json.loads(body["messages"][0]["content"])]
        return _provider_response([{"id": item_id, "translation": "譯文"} for item_id in ids])

    async def scenario() -> tuple[int, bool]:
        client = StructuredTranslationClient(
            api_key="test", model="test/model", transport=httpx.MockTransport(handler)
        )
        for page in ("page-1", "page-2"):
            request = build_request_map(page, [(f"{page}-region", "原文")])
            prompt = json.dumps([{"id": request.items[0].item_id}])
            await client.translate(request, prompt)
        await client.aclose()
        return calls, client.is_closed

    assert _run(scenario()) == (2, True)


def test_transport_http_schema_and_content_have_independent_retry_budgets() -> None:
    request = _request()
    call = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal call
        call += 1
        if call == 1:
            raise httpx.ConnectError("offline", request=http_request)
        if call == 2:
            return httpx.Response(503, text="temporary")
        if call == 3:
            return httpx.Response(200, json={"choices": []})
        if call == 4:
            return _provider_response(_items(request)[:1])
        return _provider_response(_items(request))

    async def scenario():
        async with StructuredTranslationClient(
            api_key="test",
            model="test/model",
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(transport=1, http=1, schema=1, content=1),
        ) as client:
            return await client.translate(request, "prompt")

    result = _run(scenario())
    assert call == 5
    assert result.provenance.retries == {
        "transport": 1,
        "http": 1,
        "schema": 1,
        "content": 1,
    }
    assert [attempt.category for attempt in result.provenance.attempts] == [
        "transport",
        "http",
        "schema",
        "content",
        "success",
    ]
    assert all(
        attempt.raw_response is not None
        for attempt in result.provenance.attempts
        if attempt.status_code is not None
    )


def test_policy_can_explicitly_relax_zdr_without_changing_collection_default() -> None:
    policy = ProviderPolicy(zdr=False)
    assert policy.data_collection == "deny"
    assert not policy.zdr
