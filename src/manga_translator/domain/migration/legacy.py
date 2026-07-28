"""Legacy detector adapters, isolated from canonical domain models."""

from __future__ import annotations

from uuid import UUID

from ...detector import TextRegion
from ..models import BoundingBox, RegionRevision


def region_revision_from_legacy(
    region: TextRegion,
    *,
    region_id: UUID,
    revision_id: str,
) -> RegionRevision:
    return RegionRevision(
        revision_id=revision_id,
        region_id=region_id,
        bbox=BoundingBox(
            x=float(region.x),
            y=float(region.y),
            width=float(region.w),
            height=float(region.h),
        ),
        orientation="vertical" if region.vertical else "horizontal",
        detector_score=float(region.confidence),
        source=region.source,
        raw_index=region.raw_index,
    )
