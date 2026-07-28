"""Strict JSON decoding for translation responses."""

from __future__ import annotations

import json

from .mapping import (
    MappingContractError,
    MappingIssue,
    RawResponseRef,
    RequestMap,
    ValidatedTranslationBatch,
    validate_response_items,
)


def parse_translation_response(
    response_text: str,
    request: RequestMap,
    *,
    raw_response_ref: RawResponseRef | None = None,
) -> ValidatedTranslationBatch:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise MappingContractError(
            [MappingIssue("malformed_json", {"message": str(error)})]
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"translations"}:
        raise MappingContractError(
            [MappingIssue("invalid_response_envelope", {"keys": list(payload) if isinstance(payload, dict) else []})]
        )
    if raw_response_ref is None:
        raw_response_ref = RawResponseRef.from_bytes(
            response_text.encode("utf-8"),
            media_type="application/json",
        )
    return validate_response_items(
        request,
        payload["translations"],
        raw_response_ref=raw_response_ref,
    )
