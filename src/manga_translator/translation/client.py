"""Reusable async client for strict translation requests with durable provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Self

import httpx

from ..contracts.mapping import MappingContractError, RequestMap, ValidatedTranslationBatch
from .provider import (
    ProviderEnvelopeError,
    ProviderPolicy,
    build_openrouter_payload,
    parse_provider_envelope,
    structured_output_unsupported,
)
from .schema import (
    StructuredResponseSchemaError,
    translation_json_schema,
    validate_structured_response,
)

FailureCategory = Literal["transport", "http", "schema", "content"]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    transport: int = 2
    http: int = 2
    schema: int = 1
    content: int = 1
    backoff_seconds: float = 0.0

    def limit(self, category: FailureCategory) -> int:
        return int(getattr(self, category))


@dataclass(frozen=True, slots=True)
class TranslationAttempt:
    category: Literal["success", "transport", "http", "schema", "content"]
    status_code: int | None
    raw_response: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class TranslationProvenance:
    sanitized_request: dict[str, Any]
    raw_response: str
    actual_provider: str | None
    actual_model: str | None
    usage: dict[str, Any]
    latency_ms: float
    retries: dict[str, int]
    schema_hash: str
    prompt_hash: str
    endpoint_policy: dict[str, Any]
    attempts: tuple[TranslationAttempt, ...]


@dataclass(frozen=True, slots=True)
class StructuredTranslationResult:
    batch: ValidatedTranslationBatch
    provenance: TranslationProvenance


class TranslationClientError(RuntimeError):
    def __init__(self, message: str, attempts: tuple[TranslationAttempt, ...] = ()) -> None:
        self.attempts = attempts
        super().__init__(message)


class TranslationTransportError(TranslationClientError):
    pass


class TranslationHTTPError(TranslationClientError):
    pass


class StructuredOutputUnsupported(TranslationClientError):
    pass


class TranslationSchemaError(TranslationClientError):
    pass


class TranslationContentError(TranslationClientError):
    pass


class StructuredTranslationClient:
    """One instance is one job; its connection pool is reused across page calls."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        policy: ProviderPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.policy = policy or ProviderPolicy()
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _backoff(self, retry_number: int) -> None:
        delay = self.retry_policy.backoff_seconds * (2 ** max(0, retry_number - 1))
        if delay > 0:
            await asyncio.sleep(delay)

    async def translate(
        self,
        request: RequestMap,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> StructuredTranslationResult:
        if self.is_closed:
            raise RuntimeError("structured translation client is closed")
        payload = build_openrouter_payload(
            model=self.model,
            prompt=prompt,
            policy=self.policy,
            temperature=temperature,
        )
        counters: dict[str, int] = defaultdict(int)
        attempts: list[TranslationAttempt] = []
        started = time.perf_counter()

        while True:
            try:
                response = await self._client.post(self.endpoint, json=payload)
            except httpx.TransportError as error:
                attempts.append(TranslationAttempt("transport", None, None, str(error)))
                if counters["transport"] >= self.retry_policy.transport:
                    raise TranslationTransportError(str(error), tuple(attempts)) from error
                counters["transport"] += 1
                await self._backoff(counters["transport"])
                continue

            raw_response = response.text
            if response.status_code >= 400:
                attempts.append(
                    TranslationAttempt("http", response.status_code, raw_response, "HTTP failure")
                )
                if structured_output_unsupported(response.status_code, raw_response):
                    raise StructuredOutputUnsupported(
                        "provider does not support required strict structured output",
                        tuple(attempts),
                    )
                retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
                if not retryable or counters["http"] >= self.retry_policy.http:
                    raise TranslationHTTPError(
                        f"translation provider returned HTTP {response.status_code}",
                        tuple(attempts),
                    )
                counters["http"] += 1
                await self._backoff(counters["http"])
                continue

            try:
                envelope = response.json()
                content, provider, actual_model, usage = parse_provider_envelope(envelope)
                batch = validate_structured_response(content, request)
            except (json.JSONDecodeError, ProviderEnvelopeError, StructuredResponseSchemaError) as error:
                attempts.append(
                    TranslationAttempt("schema", response.status_code, raw_response, str(error))
                )
                if counters["schema"] >= self.retry_policy.schema:
                    raise TranslationSchemaError(str(error), tuple(attempts)) from error
                counters["schema"] += 1
                await self._backoff(counters["schema"])
                continue
            except MappingContractError as error:
                attempts.append(
                    TranslationAttempt("content", response.status_code, raw_response, str(error))
                )
                if counters["content"] >= self.retry_policy.content:
                    raise TranslationContentError(str(error), tuple(attempts)) from error
                counters["content"] += 1
                await self._backoff(counters["content"])
                continue

            attempts.append(TranslationAttempt("success", response.status_code, raw_response, None))
            elapsed_ms = (time.perf_counter() - started) * 1000
            schema_bytes = json.dumps(
                translation_json_schema(), sort_keys=True, separators=(",", ":")
            ).encode()
            provenance = TranslationProvenance(
                sanitized_request=json.loads(json.dumps(payload)),
                raw_response=raw_response,
                actual_provider=provider,
                actual_model=actual_model,
                usage=usage,
                latency_ms=elapsed_ms,
                retries=dict(counters),
                schema_hash=hashlib.sha256(schema_bytes).hexdigest(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                endpoint_policy={
                    "endpoint": self.endpoint,
                    "data_collection": self.policy.data_collection,
                    "zdr": self.policy.zdr,
                    "require_parameters": self.policy.require_parameters,
                },
                attempts=tuple(attempts),
            )
            return StructuredTranslationResult(batch=batch, provenance=provenance)
