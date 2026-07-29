"""Strict structured-output schema and the post-schema exact-ID contract."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..contracts.mapping import (
    RequestMap,
    ValidatedTranslationBatch,
    validate_response_items,
)


class StructuredTranslationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    translation: str = Field(min_length=1)
    source_choice: str | None = Field(default=None, min_length=1)
    uncertainty: str | None = Field(default=None, min_length=1)
    entity_refs: list[Annotated[str, Field(min_length=1)]] | None = None

    @field_validator("entity_refs")
    @classmethod
    def entity_refs_must_be_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("entity_refs must not contain duplicates")
        return value


class StructuredTranslationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    translations: list[StructuredTranslationItem] = Field(min_length=1)


class StructuredResponseSchemaError(ValueError):
    def __init__(self, message: str, *, details: list[dict[str, Any]] | None = None) -> None:
        self.details = tuple(details or ())
        super().__init__(message)


def translation_json_schema() -> dict[str, Any]:
    return StructuredTranslationEnvelope.model_json_schema(mode="validation")


def validate_structured_response(
    content: str, request: RequestMap
) -> ValidatedTranslationBatch:
    """Validate provider JSON first, then enforce P0-03's exact-ID mapping."""

    try:
        payload = StructuredTranslationEnvelope.model_validate_json(content)
    except ValidationError as error:
        raise StructuredResponseSchemaError(
            "structured translation response does not match schema",
            details=error.errors(include_url=False),
        ) from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise StructuredResponseSchemaError("structured translation response is not JSON") from error

    expected = request.by_item_id
    raw_items = []
    for item in payload.translations:
        expected_item = expected.get(item.id)
        raw_items.append(
            {
                "id": item.id,
                "translation": item.translation,
                # The schema binds IDs. P0-03 additionally binds each accepted ID
                # to its immutable request source hash before constructing output.
                "source_sha256": expected_item.source_sha256 if expected_item is not None else "",
            }
        )
    return validate_response_items(request, raw_items)
