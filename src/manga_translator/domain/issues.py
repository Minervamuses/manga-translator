"""Machine-readable issue and stage state models."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


def ensure_json_object(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    """Reject values that cannot be represented losslessly as JSON.

    ``json.dumps`` failing at persistence time is too late: issues and entities are
    part of the canonical document contract.  Keep the public fields as ordinary
    dictionaries while validating their complete object graph eagerly.
    """

    def visit(item: Any, path: str, ancestors: set[int]) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} must not contain NaN or Infinity")
            return
        if isinstance(item, list):
            marker = id(item)
            if marker in ancestors:
                raise ValueError(f"{path} must not contain a reference cycle")
            ancestors.add(marker)
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", ancestors)
            ancestors.remove(marker)
            return
        if isinstance(item, dict):
            marker = id(item)
            if marker in ancestors:
                raise ValueError(f"{path} must not contain a reference cycle")
            ancestors.add(marker)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                visit(child, f"{path}.{key}", ancestors)
            ancestors.remove(marker)
            return
        raise ValueError(f"{path} contains non-JSON value {type(item).__name__}")

    visit(value, field_name, set())
    return value


class IssueCode(StrEnum):
    INVALID_GEOMETRY = "invalid_geometry"
    MISSING_ARTIFACT = "missing_artifact"
    SOURCE_FAILED = "source_failed"
    PIPELINE_FAILED = "pipeline_failed"
    OUTPUT_FAILED = "output_failed"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    DETECTOR_FAILED = "detector_failed"
    STYLE_FAILED = "style_failed"
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
    ORDER_UNCERTAIN = "order_uncertain"


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
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        strict=True,
    )

    code: IssueCode
    severity: IssueSeverity
    stage: StageName
    message: str = ""
    page_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    region_id: UUID | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def details_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_object(value, field_name="details")
