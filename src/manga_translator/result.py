"""Typed page and batch outcomes for the translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .detector import TextRegion

PageStatus = Literal["succeeded", "failed", "blocked"]
BatchStatus = Literal["succeeded", "partial", "failed", "blocked"]


@dataclass(frozen=True)
class ResultIssue:
    code: str
    message: str
    stage: str
    page_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.page_id is not None:
            payload["page_id"] = self.page_id
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class PageResult:
    page_id: str
    source_path: Path
    status: PageStatus
    image: np.ndarray | None = field(default=None, repr=False)
    source_image: np.ndarray | None = field(default=None, repr=False)
    regions: list[TextRegion] = field(default_factory=list, repr=False)
    ocr_results: list[str] = field(default_factory=list, repr=False)
    translations: list[str] = field(default_factory=list, repr=False)
    issues: list[ResultIssue] = field(default_factory=list)
    output_path: Path | None = None
    source_preserved: bool = False
    stage_failure: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "source_path": str(self.source_path),
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "source_preserved": self.source_preserved,
            "stage_failure": self.stage_failure,
            "region_count": len(self.regions),
        }


@dataclass
class BatchResult:
    status: BatchStatus
    pages: list[PageResult]
    issues: list[ResultIssue] = field(default_factory=list)
    manifest_path: Path | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    @property
    def failed_pages(self) -> list[PageResult]:
        return [page for page in self.pages if not page.succeeded]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "batch_result.v1",
            "status": self.status,
            "partial": self.partial,
            "page_count": len(self.pages),
            "succeeded_pages": sum(page.succeeded for page in self.pages),
            "failed_pages": len(self.failed_pages),
            "issues": [issue.to_dict() for issue in self.issues],
            "pages": [page.to_manifest() for page in self.pages],
        }


def derive_batch_status(pages: list[PageResult]) -> BatchStatus:
    if not pages:
        return "failed"
    succeeded = sum(page.succeeded for page in pages)
    if succeeded == len(pages):
        return "succeeded"
    if succeeded:
        return "partial"
    if all(page.status == "blocked" for page in pages):
        return "blocked"
    return "failed"
