"""OpenRouter structured-output payload and response envelope helpers."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .schema import translation_json_schema


class ProviderPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    data_collection: Literal["deny", "allow"] = "deny"
    zdr: bool = True
    require_parameters: Literal[True] = True


class ProviderEnvelopeError(ValueError):
    pass


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    content: str


class _Choice(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    message: _Message


class _ProviderEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    choices: list[_Choice] = Field(min_length=1, max_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)

    @field_validator("usage")
    @classmethod
    def usage_must_be_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provider usage must be finite JSON") from error
        return value


def build_openrouter_payload(
    *,
    model: str,
    prompt: str,
    policy: ProviderPolicy,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "manga_translation",
                "strict": True,
                "schema": translation_json_schema(),
            },
        },
        "provider": {
            "require_parameters": policy.require_parameters,
            "data_collection": policy.data_collection,
            "zdr": policy.zdr,
        },
    }


def parse_provider_envelope(payload: Any) -> tuple[str, str, str, dict[str, Any]]:
    try:
        envelope = _ProviderEnvelope.model_validate(payload)
    except ValidationError as error:
        raise ProviderEnvelopeError("provider response envelope is invalid") from error
    return (
        envelope.choices[0].message.content,
        envelope.provider,
        envelope.model,
        envelope.usage,
    )


def structured_output_unsupported(status_code: int, response_text: str) -> bool:
    if status_code not in {400, 404, 422}:
        return False
    lowered = response_text.lower()
    signals = ("response_format", "json_schema", "structured output", "require_parameters")
    return any(signal in lowered for signal in signals)
