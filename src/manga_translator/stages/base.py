"""Typed contracts shared by all pipeline stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..domain.issues import StageName
from ..domain.models import ArtifactRef


@dataclass(frozen=True)
class ArtifactPayload:
    data: bytes
    media_type: str
    role: str


@dataclass(frozen=True)
class StageInputs:
    upstream: Mapping[StageName, tuple[ArtifactRef, ...]]


@dataclass(frozen=True)
class StageOutputs:
    artifacts: tuple[ArtifactPayload, ...]


@dataclass(frozen=True)
class FingerprintDependencies:
    model_hashes: tuple[str, ...] = ()
    font_hashes: tuple[str, ...] = ()
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    preprocess_revision: str = ""
    prompt_revision: str = ""
    schema_revision: str = "1.0"
    glossary_revision: str = ""
    entity_revision: str = ""


StageFunction = Callable[["StageContext", StageInputs], StageOutputs]


@dataclass(frozen=True)
class StageSpec:
    name: StageName
    dependencies: tuple[StageName, ...]
    run: StageFunction
    code_revision: str
    config_keys: tuple[str, ...] = ()
    fingerprint_dependencies: FingerprintDependencies = field(
        default_factory=FingerprintDependencies
    )


class StageContext:
    def __init__(
        self,
        *,
        job_id: str,
        page_id: str,
        stage: StageName,
        fingerprint: str,
        config: Mapping[str, Any],
        raw_response: Callable[[str, Callable[[], bytes]], bytes],
    ) -> None:
        self.job_id = job_id
        self.page_id = page_id
        self.stage = stage
        self.fingerprint = fingerprint
        self.config = config
        self._raw_response = raw_response

    def get_or_fetch_raw_response(
        self, request_key: str, fetch: Callable[[], bytes]
    ) -> bytes:
        """Replay a durable provider response, fetching only on a cache miss."""

        if not request_key:
            raise ValueError("request_key must not be empty")
        return self._raw_response(request_key, fetch)
