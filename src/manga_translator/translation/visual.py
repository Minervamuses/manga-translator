"""Opt-in, trigger-bound and privacy-auditable visual-context escalation."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import cv2
import numpy as np

from ..config import VisualContextConfig
from ..contracts.mapping import request_map_from_ids, source_sha256
from .provider import ProviderPolicy, build_openrouter_payload
from .schema import validate_structured_response
from .units import TranslationUnit


class VisualTrigger(StrEnum):
    ORDER_UNCERTAIN = "order_uncertain"
    OCR_CANDIDATE_AMBIGUITY = "ocr_candidate_ambiguity"
    SPEAKER_ENTITY_AMBIGUITY = "speaker_entity_ambiguity"


@dataclass(frozen=True, slots=True)
class VisualModelProfile:
    model: str
    benchmark_profile: str

    def __post_init__(self) -> None:
        if not self.model or not self.benchmark_profile:
            raise ValueError("visual model must come from a named benchmark profile")


@dataclass(frozen=True, slots=True)
class VisualProviderRequest:
    payload: dict[str, Any]
    image_png: bytes
    unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisualProviderResponse:
    content: str
    provider: str
    model: str
    usage: dict[str, Any]
    cost: float | None = None


VisualProvider = Callable[[VisualProviderRequest], Awaitable[VisualProviderResponse]]


@dataclass(frozen=True, slots=True)
class VisualRequestManifest:
    status: Literal["not_requested", "blocked", "succeeded", "failed"]
    sent_image: bool
    unit_ids: tuple[str, ...]
    triggers: dict[str, tuple[str, ...]]
    image_sha256: str | None
    image_dimensions: tuple[int, int] | None
    provider: str | None
    model: str | None
    benchmark_profile: str | None
    data_collection: Literal["deny", "allow"]
    zdr: bool
    cost: float | None
    usage: dict[str, Any]
    issue: str | None = None


@dataclass(frozen=True, slots=True)
class VisualEscalationResult:
    translations: dict[str, str]
    manifest: VisualRequestManifest


def _encode_overlay(
    image: np.ndarray, units: Sequence[TranslationUnit], max_side: int
) -> tuple[bytes, tuple[int, int]]:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] in {3, 4}:
        canvas = image[:, :, :3].copy()
    else:
        raise ValueError("visual context image must be grayscale, BGR, or BGRA")
    height, width = canvas.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        canvas = cv2.resize(
            canvas,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    overlay_height, overlay_width = canvas.shape[:2]
    for unit in units:
        box = unit.normalized_bbox
        left = round(box.x * overlay_width)
        top = round(box.y * overlay_height)
        right = round((box.x + box.width) * overlay_width)
        bottom = round((box.y + box.height) * overlay_height)
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            unit.request_item_id,
            (left, max(12, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to encode visual context overlay")
    return encoded.tobytes(), (overlay_width, overlay_height)


def _manifest(
    *,
    status: Literal["not_requested", "blocked", "succeeded", "failed"],
    policy: ProviderPolicy,
    originals: Mapping[str, str],
    triggers: Mapping[str, Sequence[VisualTrigger]],
    **updates: Any,
) -> VisualRequestManifest:
    defaults: dict[str, Any] = {
        "sent_image": False,
        "unit_ids": tuple(unit_id for unit_id in originals if triggers.get(unit_id)),
        "triggers": {
            unit_id: tuple(trigger.value for trigger in unit_triggers)
            for unit_id, unit_triggers in triggers.items()
            if unit_triggers
        },
        "image_sha256": None,
        "image_dimensions": None,
        "provider": None,
        "model": None,
        "benchmark_profile": None,
        "data_collection": policy.data_collection,
        "zdr": policy.zdr,
        "cost": None,
        "usage": {},
        "issue": None,
    }
    defaults.update(updates)
    return VisualRequestManifest(status=status, **defaults)


async def escalate_visual_context(
    *,
    image: np.ndarray,
    units: Sequence[TranslationUnit],
    original_translations: Mapping[str, str],
    triggers: Mapping[str, Sequence[VisualTrigger]],
    config: VisualContextConfig,
    model_profile: VisualModelProfile | None,
    provider: VisualProvider,
    endpoint_supports_zdr: bool,
    data_collection: Literal["deny", "allow"] = "deny",
) -> VisualEscalationResult:
    """Update only explicitly uncertain units; every failure returns original translations."""

    originals = dict(original_translations)
    policy = ProviderPolicy(data_collection=data_collection, zdr=config.require_zdr)
    uncertain = tuple(unit for unit in units if triggers.get(unit.request_item_id))
    if not config.enabled or not uncertain:
        return VisualEscalationResult(
            originals,
            _manifest(
                status="not_requested",
                policy=policy,
                originals=originals,
                triggers=triggers,
            ),
        )
    if model_profile is None:
        return VisualEscalationResult(
            originals,
            _manifest(
                status="blocked",
                policy=policy,
                originals=originals,
                triggers=triggers,
                issue="no benchmark-selected visual model profile",
            ),
        )
    if config.require_zdr and not endpoint_supports_zdr and not config.allow_non_zdr:
        return VisualEscalationResult(
            originals,
            _manifest(
                status="blocked",
                policy=policy,
                originals=originals,
                triggers=triggers,
                model=model_profile.model,
                benchmark_profile=model_profile.benchmark_profile,
                issue="endpoint does not support required ZDR",
            ),
        )
    effective_zdr = config.require_zdr and endpoint_supports_zdr
    policy = ProviderPolicy(data_collection=data_collection, zdr=effective_zdr)
    image_png, dimensions = _encode_overlay(image, uncertain, config.max_image_side)
    image_hash = hashlib.sha256(image_png).hexdigest()
    ids = [unit.request_item_id for unit in uncertain]
    hashes = [source_sha256(unit.ocr_nfc) for unit in uncertain]
    request_map = request_map_from_ids(ids, hashes, request_id="visual")
    prompt = json.dumps(
        {
            "task": "resolve only the listed uncertain manga translation units",
            "units": [
                {
                    "id": unit.request_item_id,
                    "source": unit.ocr_nfc,
                    "triggers": [item.value for item in triggers[unit.request_item_id]],
                }
                for unit in uncertain
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = build_openrouter_payload(
        model=model_profile.model, prompt=prompt, policy=policy, temperature=0.0
    )
    payload["messages"][0]["content"] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
            },
        },
    ]
    response: VisualProviderResponse | None = None
    try:
        response = await provider(VisualProviderRequest(payload, image_png, tuple(ids)))
        batch = validate_structured_response(response.content, request_map)
    except Exception as error:  # noqa: BLE001 - escalation failure must preserve valid text
        return VisualEscalationResult(
            originals,
            _manifest(
                status="failed",
                policy=policy,
                originals=originals,
                triggers=triggers,
                sent_image=True,
                image_sha256=image_hash,
                image_dimensions=dimensions,
                provider=response.provider if response is not None else None,
                model=response.model if response is not None else model_profile.model,
                benchmark_profile=model_profile.benchmark_profile,
                cost=response.cost if response is not None else None,
                usage=response.usage if response is not None else {},
                issue=str(error),
            ),
        )
    updated = dict(originals)
    for unit in uncertain:
        updated[unit.request_item_id] = batch.responses[ids.index(unit.request_item_id)].translation
    return VisualEscalationResult(
        updated,
        _manifest(
            status="succeeded",
            policy=policy,
            originals=originals,
            triggers=triggers,
            sent_image=True,
            image_sha256=image_hash,
            image_dimensions=dimensions,
            provider=response.provider,
            model=response.model,
            benchmark_profile=model_profile.benchmark_profile,
            cost=response.cost,
            usage=response.usage,
        ),
    )
