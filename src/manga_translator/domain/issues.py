"""Machine-readable issue and stage state models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IssueCode(StrEnum):
    INVALID_GEOMETRY = "invalid_geometry"
    MISSING_ARTIFACT = "missing_artifact"
    SOURCE_FAILED = "source_failed"
    PIPELINE_FAILED = "pipeline_failed"
    OUTPUT_FAILED = "output_failed"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    DETECTOR_FAILED = "detector_failed"
    OCR_FAILED = "ocr_failed"
    OCR_REJECTED = "ocr_rejected"
    TRANSLATION_FAILED = "translation_failed"
    TRANSLATION_REJECTED = "translation_rejected"
    LAYOUT_FAILED = "layout_failed"
    LAYOUT_REJECTED = "layout_rejected"
    RENDER_FAILED = "render_failed"
    ENCODE_FAILED = "encode_failed"
    CACHE_INVALIDATED = "cache_invalidated"
    STAGE_INTERRUPTED = "stage_interrupted"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class StageName(StrEnum):
    SOURCE = "source"
    DETECT = "detect"
    STYLE = "style"
    SAFE_REGION = "safe_region"
    OCR = "ocr"
    ORDER = "order"
    TRANSLATE = "translate"
    LAYOUT = "layout"
    INPAINT_RENDER = "inpaint_render"
    ENCODE = "encode"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    INTERRUPTED = "interrupted"


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: IssueCode
    severity: IssueSeverity
    stage: StageName
    message: str = ""
    page_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    region_id: UUID | None = None
    details: dict[str, Any] = Field(default_factory=dict)
