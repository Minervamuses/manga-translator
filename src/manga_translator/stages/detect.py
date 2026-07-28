"""Canonical detector geometry projection used by the detect stage."""

from __future__ import annotations

import json

import numpy as np

from ..detector import DetectionResult
from .base import ArtifactPayload, StageOutputs

DETECTOR_GEOMETRY_MEDIA_TYPE = "application/vnd.manga-translator.detector-geometry+json"


def detection_geometry_output(detection: DetectionResult) -> StageOutputs:
    """Serialize geometry and lineage without embedding raster mask bytes."""

    payload = {
        "groups": [
            {
                "bbox": list(group.geometry_bbox or tuple(float(value) for value in group.bbox)),
                "group_id": group.id,
                "mask_empty": group.mask is None or not bool(np.any(group.mask)),
                "mask_sources": [source.__dict__ for source in group.mask_sources],
                "region_ids": list(group.region_ids),
            }
            for group in detection.groups
        ],
        "issues": [
            {"code": issue.code, "message": issue.message, "details": issue.details}
            for issue in detection.issues
        ],
        "regions": [
            {
                "angle_degrees": region.angle_degrees,
                "font_size_hint": region.font_size_hint,
                "id": region.id,
                "line_polygons": region.line_polygons,
                "mask_empty": region.local_mask is None or not bool(np.any(region.local_mask)),
                "mask_source": region.mask_source.__dict__ if region.mask_source else None,
                "page_bbox": region.page_bbox,
                "raster_bbox": region.bbox,
                "raw_index": region.raw_index,
                "source": region.source,
            }
            for region in detection.regions_post
        ],
        "schema_version": "detector_geometry.v1",
    }
    return StageOutputs(
        (
            ArtifactPayload(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                DETECTOR_GEOMETRY_MEDIA_TYPE,
                "geometry",
            ),
        )
    )
