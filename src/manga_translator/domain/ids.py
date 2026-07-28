"""Stable page, region, and revision identifiers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from uuid import UUID, uuid4

from .models import ArtifactRef, BoundingBox, Polygon


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def page_id_from_bytes(original_bytes: bytes) -> str:
    """Return the immutable identity of the exact source-page bytes."""

    return hashlib.sha256(original_bytes).hexdigest()


def new_region_id() -> UUID:
    """Create a persistent identity for a newly confirmed logical region."""

    return uuid4()


def canonical_geometry_bytes(
    bbox: BoundingBox,
    polygon: Polygon | None = None,
    line_polygons: Sequence[Polygon] = (),
    angle_degrees: float = 0.0,
    orientation: str = "unknown",
) -> bytes:
    if not math.isfinite(angle_degrees):
        raise ValueError("angle_degrees must be finite")
    if orientation not in {"horizontal", "vertical", "rotated", "unknown"}:
        raise ValueError("orientation is unsupported")
    payload = {
        "angle_degrees": 0.0 if angle_degrees == 0.0 else angle_degrees,
        "bbox": bbox.model_dump(mode="json"),
        "line_polygons": [item.model_dump(mode="json") for item in line_polygons],
        "orientation": orientation,
        "polygon": polygon.model_dump(mode="json") if polygon is not None else None,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def revision_id_for(
    *,
    page_id: str,
    detector_fingerprint: str,
    bbox: BoundingBox,
    polygon: Polygon | None = None,
    line_polygons: Sequence[Polygon] = (),
    angle_degrees: float = 0.0,
    orientation: str = "unknown",
    mask_refs: Sequence[ArtifactRef] = (),
) -> str:
    """Hash every input that defines a detector revision, excluding its durable identity."""

    _validate_sha256(page_id, "page_id")
    _validate_sha256(detector_fingerprint, "detector_fingerprint")
    digest = hashlib.sha256()
    for label, value in (
        (b"page", page_id.encode("ascii")),
        (b"detector", detector_fingerprint.encode("ascii")),
        (
            b"geometry",
            canonical_geometry_bytes(
                bbox,
                polygon,
                line_polygons,
                angle_degrees,
                orientation,
            ),
        ),
        (
            b"masks",
            json.dumps([ref.sha256 for ref in mask_refs], separators=(",", ":")).encode("ascii"),
        ),
    ):
        digest.update(len(label).to_bytes(2, "big"))
        digest.update(label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def dhash_distance(left: int, right: int, *, bits: int = 64) -> float:
    """Return normalized perceptual-hash distance in the closed interval [0, 1]."""

    if left < 0 or right < 0 or bits <= 0 or left.bit_length() > bits or right.bit_length() > bits:
        raise ValueError("dHash values must be non-negative and fit within bits")
    return (left ^ right).bit_count() / bits
