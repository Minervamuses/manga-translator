from __future__ import annotations

import asyncio
import json
import math
from uuid import UUID

import cv2
import numpy as np
import pytest

from manga_translator.config import VisualContextConfig
from manga_translator.translation.units import NormalizedBoundingBox, TranslationUnit
from manga_translator.translation.visual import (
    VisualModelProfile,
    VisualProviderResponse,
    VisualTrigger,
    escalate_visual_context,
)


def _unit(number: int, request_id: str, *, uncertain: bool = False) -> TranslationUnit:
    return TranslationUnit(
        request_item_id=request_id,
        region_id=UUID(int=number),
        panel_id="panel",
        order=number - 1,
        normalized_bbox=NormalizedBoundingBox(
            x=0.1 + (number - 1) * 0.5, y=0.2, width=0.25, height=0.4
        ),
        orientation="vertical",
        kind="dialogue",
        ocr_raw="原文",
        ocr_nfc="原文",
        confidence=0.5,
        confidence_kind="calibrated",
        order_uncertain=uncertain,
    )


def _run(coro):
    return asyncio.run(coro)


def test_disabled_visual_context_never_calls_provider_or_sends_image() -> None:
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("disabled visual context must not call provider")

    originals = {"u0001": "第一次有效譯文"}
    result = _run(
        escalate_visual_context(
            image=np.full((100, 100, 3), 255, dtype=np.uint8),
            units=(_unit(1, "u0001", uncertain=True),),
            original_translations=originals,
            triggers={"u0001": (VisualTrigger.ORDER_UNCERTAIN,)},
            config=VisualContextConfig(),
            model_profile=None,
            provider=provider,
            endpoint_supports_zdr=False,
        )
    )
    assert calls == 0
    assert result.translations == originals
    assert result.manifest.status == "not_requested"
    assert not result.manifest.sent_image


def test_triggered_visual_request_uploads_low_resolution_overlay_and_maps_exact_id() -> None:
    captured = []
    units = (_unit(1, "u0001", uncertain=True), _unit(2, "u0002"))

    async def provider(request):
        captured.append(request)
        return VisualProviderResponse(
            content=json.dumps(
                {"translations": [{"id": "u0001", "translation": "視覺修正版"}]},
                ensure_ascii=False,
            ),
            provider="fixture-provider",
            model="bench/model",
            usage={"image_tokens": 42},
            cost=0.003,
        )

    result = _run(
        escalate_visual_context(
            image=np.full((800, 1600, 3), 255, dtype=np.uint8),
            units=units,
            original_translations={"u0001": "第一次譯文", "u0002": "不應改動"},
            triggers={"u0001": (VisualTrigger.SPEAKER_ENTITY_AMBIGUITY,)},
            config=VisualContextConfig(enabled=True, max_image_side=400),
            model_profile=VisualModelProfile("bench/model", "translation-visual-v1"),
            provider=provider,
            endpoint_supports_zdr=True,
        )
    )

    assert len(captured) == 1
    request = captured[0]
    assert request.unit_ids == ("u0001",)
    decoded = cv2.imdecode(np.frombuffer(request.image_png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (200, 400)
    assert request.payload["response_format"]["json_schema"]["strict"] is True
    assert request.payload["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    content = request.payload["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "image_url"]
    assert result.translations == {"u0001": "視覺修正版", "u0002": "不應改動"}
    assert result.manifest.status == "succeeded"
    assert result.manifest.sent_image
    assert result.manifest.image_dimensions == (400, 200)
    assert len(result.manifest.image_sha256 or "") == 64
    assert result.manifest.provider == "fixture-provider"
    assert result.manifest.model == "bench/model"
    assert result.manifest.cost == 0.003


def test_visual_response_failure_never_overwrites_valid_text_or_erases_original() -> None:
    async def provider(_request):
        return VisualProviderResponse(
            content='{"translations":[{"id":"wrong","translation":"錯誤"}]}',
            provider="fixture",
            model="bench/model",
            usage={},
        )

    originals = {"u0001": "第一次有效譯文"}
    result = _run(
        escalate_visual_context(
            image=np.full((100, 100), 255, dtype=np.uint8),
            units=(_unit(1, "u0001", uncertain=True),),
            original_translations=originals,
            triggers={"u0001": (VisualTrigger.OCR_CANDIDATE_AMBIGUITY,)},
            config=VisualContextConfig(enabled=True),
            model_profile=VisualModelProfile("bench/model", "profile"),
            provider=provider,
            endpoint_supports_zdr=True,
        )
    )
    assert result.translations == originals
    assert result.manifest.status == "failed"
    assert result.manifest.sent_image


def test_semantically_invalid_visual_response_preserves_first_valid_translation() -> None:
    async def provider(_request):
        return VisualProviderResponse(
            content='{"translations":[{"id":"u0001","translation":"どうしたの"}]}',
            provider="fixture",
            model="bench/model",
            usage={},
        )

    originals = {"u0001": "第一次有效譯文"}
    unit = _unit(1, "u0001", uncertain=True).model_copy(
        update={"ocr_raw": "どうしたの", "ocr_nfc": "どうしたの"}
    )
    result = _run(
        escalate_visual_context(
            image=np.full((100, 100), 255, dtype=np.uint8),
            units=(unit,),
            original_translations=originals,
            triggers={"u0001": (VisualTrigger.OCR_CANDIDATE_AMBIGUITY,)},
            config=VisualContextConfig(enabled=True),
            model_profile=VisualModelProfile("bench/model", "profile"),
            provider=provider,
            endpoint_supports_zdr=True,
        )
    )

    assert result.translations == originals
    assert result.manifest.status == "failed"
    assert "semantic validation" in (result.manifest.issue or "")


def test_local_image_failure_is_audited_without_claiming_an_upload() -> None:
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid local image must not be uploaded")

    originals = {"u0001": "第一次有效譯文"}
    result = _run(
        escalate_visual_context(
            image=np.zeros((0, 0), dtype=np.uint8),
            units=(_unit(1, "u0001", uncertain=True),),
            original_translations=originals,
            triggers={"u0001": (VisualTrigger.ORDER_UNCERTAIN,)},
            config=VisualContextConfig(enabled=True),
            model_profile=VisualModelProfile("bench/model", "profile"),
            provider=provider,
            endpoint_supports_zdr=True,
        )
    )

    assert calls == 0
    assert result.translations == originals
    assert result.manifest.status == "failed"
    assert not result.manifest.sent_image
    assert result.manifest.image_sha256 is None


def test_visual_provider_identity_and_accounting_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        VisualProviderResponse("{}", "provider", "model", {"tokens": math.nan})
    with pytest.raises(ValueError, match="non-negative"):
        VisualProviderResponse("{}", "provider", "model", {}, cost=-0.01)

    async def wrong_model(_request):
        return VisualProviderResponse(
            content='{"translations":[{"id":"u0001","translation":"修正版"}]}',
            provider="fixture",
            model="other/model",
            usage={},
        )

    originals = {"u0001": "原譯"}
    result = _run(
        escalate_visual_context(
            image=np.full((100, 100), 255, dtype=np.uint8),
            units=(_unit(1, "u0001", uncertain=True),),
            original_translations=originals,
            triggers={"u0001": (VisualTrigger.ORDER_UNCERTAIN,)},
            config=VisualContextConfig(enabled=True),
            model_profile=VisualModelProfile("bench/model", "profile"),
            provider=wrong_model,
            endpoint_supports_zdr=True,
        )
    )
    assert result.translations == originals
    assert result.manifest.status == "failed"
    assert result.manifest.model == "other/model"


def test_zdr_requirement_blocks_unsupported_endpoint_unless_explicitly_relaxed() -> None:
    calls = []

    async def provider(request):
        calls.append(request)
        return VisualProviderResponse(
            content='{"translations":[{"id":"u0001","translation":"修正版"}]}',
            provider="fixture",
            model="bench/model",
            usage={},
        )

    arguments = {
        "image": np.full((100, 100), 255, dtype=np.uint8),
        "units": (_unit(1, "u0001", uncertain=True),),
        "original_translations": {"u0001": "原譯"},
        "triggers": {"u0001": (VisualTrigger.ORDER_UNCERTAIN,)},
        "model_profile": VisualModelProfile("bench/model", "profile"),
        "provider": provider,
        "endpoint_supports_zdr": False,
    }
    blocked = _run(escalate_visual_context(config=VisualContextConfig(enabled=True), **arguments))
    relaxed = _run(
        escalate_visual_context(
            config=VisualContextConfig(enabled=True, allow_non_zdr=True), **arguments
        )
    )

    assert blocked.manifest.status == "blocked"
    assert not blocked.manifest.sent_image
    assert relaxed.manifest.status == "succeeded"
    assert not relaxed.manifest.zdr
    assert len(calls) == 1


def test_no_trigger_or_missing_benchmark_profile_never_uploads() -> None:
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        raise AssertionError

    base = {
        "image": np.full((100, 100), 255, dtype=np.uint8),
        "units": (_unit(1, "u0001"),),
        "original_translations": {"u0001": "原譯"},
        "config": VisualContextConfig(enabled=True),
        "provider": provider,
        "endpoint_supports_zdr": True,
    }
    no_trigger = _run(escalate_visual_context(triggers={}, model_profile=None, **base))
    no_profile = _run(
        escalate_visual_context(
            triggers={"u0001": (VisualTrigger.ORDER_UNCERTAIN,)},
            model_profile=None,
            **base,
        )
    )
    assert calls == 0
    assert no_trigger.manifest.status == "not_requested"
    assert no_profile.manifest.status == "blocked"
