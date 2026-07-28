"""Typed page and batch outcomes for the translation pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .contracts.mapping import normalize_mapping_chain
from .detector import TextGroup, TextRegion

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


@dataclass(frozen=True)
class GroupMappingSnapshot:
    """Immutable manifest view of one group's mapping progress."""

    group_id: str
    region_ids: tuple[str, ...]
    group_status: str
    translation_valid: bool
    skip_reason: str
    duplicate_of: str | None
    chain: dict[str, Any]

    @classmethod
    def from_group(cls, group: TextGroup) -> GroupMappingSnapshot:
        return cls(
            group_id=group.id,
            region_ids=tuple(group.region_ids),
            group_status=group.status,
            translation_valid=group.translation_valid,
            skip_reason=group.skip_reason,
            duplicate_of=group.duplicate_of,
            chain=deepcopy(normalize_mapping_chain(group.mapping_chain)),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "region_ids": list(self.region_ids),
            "group_status": self.group_status,
            "translation_valid": self.translation_valid,
            "skip_reason": self.skip_reason,
            "duplicate_of": self.duplicate_of,
            "chain": deepcopy(normalize_mapping_chain(self.chain)),
        }


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
    mapping_chains: list[GroupMappingSnapshot] = field(default_factory=list)
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
            "mapping_chains": [chain.to_manifest() for chain in self.mapping_chains],
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
