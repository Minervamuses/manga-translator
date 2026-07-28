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
class ArtifactContract:
    role: str
    media_type: str

    def __post_init__(self) -> None:
        if not self.role or not self.media_type:
            raise ValueError("artifact contract role and media_type must not be empty")


@dataclass(frozen=True)
class ArtifactSetContract:
    required: tuple[ArtifactContract, ...]
    additional_media_types: tuple[str, ...] = ()

    def validate_payloads(self, artifacts: tuple[ArtifactPayload, ...]) -> None:
        if len(artifacts) < len(self.required):
            raise ValueError("stage returned fewer artifacts than its typed output contract")
        for index, expected in enumerate(self.required):
            actual = artifacts[index]
            if actual.role != expected.role or actual.media_type != expected.media_type:
                raise ValueError(
                    "stage output contract mismatch at index "
                    f"{index}: expected {expected.role}/{expected.media_type}, "
                    f"got {actual.role}/{actual.media_type}"
                )
        for actual in artifacts[len(self.required) :]:
            if actual.media_type not in self.additional_media_types:
                raise ValueError(
                    f"stage returned undeclared additional media type: {actual.media_type}"
                )

    def validate_refs(self, artifacts: tuple[ArtifactRef, ...], *, dependency: StageName) -> None:
        if len(artifacts) < len(self.required):
            raise ValueError(
                f"stage input {dependency.value} has fewer artifacts than its typed contract"
            )
        for index, expected in enumerate(self.required):
            if artifacts[index].media_type != expected.media_type:
                raise ValueError(
                    f"stage input {dependency.value} media type mismatch at index {index}"
                )
        for actual in artifacts[len(self.required) :]:
            if actual.media_type not in self.additional_media_types:
                raise ValueError(
                    f"stage input {dependency.value} has undeclared media type: "
                    f"{actual.media_type}"
                )


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
    input_contracts: Mapping[StageName, ArtifactSetContract] = field(default_factory=dict)
    output_contract: ArtifactSetContract | None = None
    fingerprint_dependencies: FingerprintDependencies = field(
        default_factory=FingerprintDependencies
    )

    def __post_init__(self) -> None:
        if not self.code_revision:
            raise ValueError("stage code_revision must not be empty")
        if set(self.input_contracts) - set(self.dependencies):
            raise ValueError("stage input contracts may only describe declared dependencies")


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
