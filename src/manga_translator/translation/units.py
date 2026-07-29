"""Stable translation request units derived from a PageDocument."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import Field

from ..domain.models import DomainModel, OCRCandidate, PageDocument
from ..order.panels import PanelCandidate
from ..reading_order import OrderRegion, ReadingOrderIssue, resolve_reading_order


class NormalizedBoundingBox(DomainModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class TranslationUnit(DomainModel):
    request_item_id: str = Field(pattern=r"^u[0-9]{4,}$")
    region_id: UUID
    panel_id: str | None = None
    order: int = Field(ge=0)
    normalized_bbox: NormalizedBoundingBox
    orientation: Literal["horizontal", "vertical", "rotated", "unknown"]
    kind: Literal["dialogue", "caption", "sfx", "other", "unknown"]
    ocr_raw: str
    ocr_nfc: str
    candidates: tuple[OCRCandidate, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_kind: Literal["model", "calibrated", "heuristic", "ensemble", "unknown"]
    order_uncertain: bool = False


@dataclass(frozen=True, slots=True)
class TranslationUnitBuildResult:
    units: tuple[TranslationUnit, ...]
    order_confidence: float
    order_uncertain: bool
    used_manual_override: bool
    issues: tuple[ReadingOrderIssue, ...]


def _manual_panels(document: PageDocument) -> tuple[PanelCandidate, ...]:
    return tuple(
        PanelCandidate(
            panel_id=panel.panel_id,
            x=panel.bbox.x,
            y=panel.bbox.y,
            width=panel.bbox.width,
            height=panel.bbox.height,
            confidence=1.0,
            source="manual",
        )
        for panel in sorted(document.panel_overrides, key=lambda item: item.order)
    )


def build_translation_units(
    document: PageDocument,
    *,
    panels: Sequence[PanelCandidate] = (),
) -> TranslationUnitBuildResult:
    """Build exact-ID units while keeping OCR content attached to persistent region IDs."""

    active_identities = tuple(
        identity for identity in document.region_identities if identity.is_active
    )
    active_region_ids = {identity.region_id for identity in active_identities}
    active_revisions = {
        identity.region_id: next(
            revision
            for revision in document.region_revisions
            if revision.revision_id == identity.active_revision_id
        )
        for identity in active_identities
    }
    order_regions = tuple(
        OrderRegion(
            identity.region_id,
            active_revisions[identity.region_id].bbox,
            active_revisions[identity.region_id].orientation,
        )
        for identity in active_identities
    )
    effective_panels = _manual_panels(document) or tuple(panels)
    order = resolve_reading_order(
        order_regions,
        panels=effective_panels,
        manual_overrides=tuple(
            override
            for override in document.reading_order_overrides
            if override.region_id in active_region_ids
        ),
    )
    ocr_by_revision = {}
    for record in document.ocr_records:
        key = (record.region_id, record.revision_id)
        if key in ocr_by_revision:
            raise ValueError("duplicate OCR record for a region revision")
        ocr_by_revision[key] = record
    units: list[TranslationUnit] = []
    for ordered in order.regions:
        revision = active_revisions[ordered.region_id]
        record = ocr_by_revision.get((ordered.region_id, revision.revision_id))
        candidates = record.candidates if record is not None else ()
        selected: OCRCandidate | None = None
        if record is not None and record.selected_index is not None:
            selected = candidates[record.selected_index]
        elif candidates:
            selected = max(candidates, key=lambda item: item.confidence)
        raw = selected.raw_text if selected is not None else ""
        units.append(
            TranslationUnit(
                request_item_id=f"u{ordered.order + 1:04d}",
                region_id=ordered.region_id,
                panel_id=ordered.panel_id,
                order=ordered.order,
                normalized_bbox=NormalizedBoundingBox(
                    x=revision.bbox.x / document.source.width,
                    y=revision.bbox.y / document.source.height,
                    width=revision.bbox.width / document.source.width,
                    height=revision.bbox.height / document.source.height,
                ),
                orientation=revision.orientation,
                kind=revision.kind,
                ocr_raw=raw,
                ocr_nfc=unicodedata.normalize("NFC", raw),
                candidates=candidates,
                confidence=selected.confidence if selected is not None else 0.0,
                confidence_kind=selected.confidence_kind if selected is not None else "unknown",
                order_uncertain=order.order_uncertain,
            )
        )
    return TranslationUnitBuildResult(
        units=tuple(units),
        order_confidence=order.confidence,
        order_uncertain=order.order_uncertain,
        used_manual_override=order.used_manual_override,
        issues=order.issues,
    )
