"""Deterministic, artifact-backed state for the v0.3.2 stage adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..detector import DetectionResult, DetectorIssue, MaskSource, TextGroup, TextRegion
from ..domain.issues import StageName
from ..domain.models import ArtifactRef
from .base import ArtifactPayload, StageOutputs

STATE_SCHEMA = "pipeline_stage_state.v3"
STATE_MEDIA_TYPE = "application/vnd.manga-translator.pipeline-state+json"
MASK_MEDIA_TYPE = "image/png"


@dataclass(frozen=True)
class PipelineStageState:
    detection: DetectionResult
    extras: dict[str, Any]


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _encode_mask(mask: np.ndarray) -> bytes:
    material = np.asarray(mask)
    if material.ndim not in {2, 3} or material.size == 0:
        raise ValueError("stage mask must be a non-empty 2D/3D array")
    if material.dtype != np.uint8:
        material = np.clip(material, 0, 255).astype(np.uint8)
    encoded, buffer = cv2.imencode(".png", material)
    if not encoded:
        raise ValueError("stage mask could not be encoded as PNG")
    return buffer.tobytes()


def _mask_reference(
    mask: np.ndarray | None,
    payloads: dict[str, bytes],
) -> str | None:
    if mask is None:
        return None
    payload = _encode_mask(mask)
    sha256 = hashlib.sha256(payload).hexdigest()
    payloads.setdefault(sha256, payload)
    return sha256


def _region_payload(region: TextRegion, payloads: dict[str, bytes]) -> dict[str, Any]:
    return {
        "angle_degrees": region.angle_degrees,
        "candidate_duplicate": region.candidate_duplicate,
        "confidence": region.confidence,
        "detection_input_size": region.detection_input_size,
        "font_size_hint": region.font_size_hint,
        "group_id": region.group_id,
        "h": region.h,
        "id": region.id,
        "local_mask": _mask_reference(region.local_mask, payloads),
        "line_polygons": [
            [[float(x), float(y)] for x, y in polygon]
            for polygon in region.line_polygons
        ],
        "mask_bbox": list(region.mask_bbox) if region.mask_bbox is not None else None,
        "mask_source": (
            {
                "detection_input_size": region.mask_source.detection_input_size,
                "detector_pass": region.mask_source.detector_pass,
                "page_to_local_affine": list(region.mask_source.page_to_local_affine),
                "raw_index": region.mask_source.raw_index,
                "source": region.mask_source.source,
                "source_region_id": region.mask_source.source_region_id,
            }
            if region.mask_source is not None
            else None
        ),
        "page_bbox": list(region.page_bbox) if region.page_bbox is not None else None,
        "raw_index": region.raw_index,
        "source": region.source,
        "vertical": region.vertical,
        "w": region.w,
        "x": region.x,
        "y": region.y,
    }


def _group_payload(group: TextGroup, payloads: dict[str, bytes]) -> dict[str, Any]:
    return {
        "bbox": list(group.bbox),
        "duplicate_of": group.duplicate_of,
        "id": group.id,
        "geometry_bbox": (
            list(group.geometry_bbox) if group.geometry_bbox is not None else None
        ),
        "layout_bbox": list(group.layout_bbox) if group.layout_bbox is not None else None,
        "layout_info": group.layout_info,
        "layout_mode": group.layout_mode,
        "mapping_chain": group.mapping_chain,
        "mapping_region_key": group.mapping_region_key,
        "mask": _mask_reference(group.mask, payloads),
        "mask_sources": [
            {
                "detection_input_size": source.detection_input_size,
                "detector_pass": source.detector_pass,
                "page_to_local_affine": list(source.page_to_local_affine),
                "raw_index": source.raw_index,
                "source": source.source,
                "source_region_id": source.source_region_id,
            }
            for source in group.mask_sources
        ],
        "ocr_candidates": group.ocr_candidates,
        "ocr_confidence": group.ocr_confidence,
        "ocr_source": group.ocr_source,
        "ocr_text": group.ocr_text,
        "ocr_text_norm": group.ocr_text_norm,
        "region_ids": list(group.region_ids),
        "rendered_direction": group.rendered_direction,
        "rendered_font_size": group.rendered_font_size,
        "skip_reason": group.skip_reason,
        "sort_key": list(group.sort_key),
        "status": group.status,
        "stable_group_key": _stable_group_key(group),
        "translation": group.translation,
        "translation_valid": group.translation_valid,
        "vertical": group.vertical,
    }


def _stable_group_key(group: TextGroup) -> str:
    material = _canonical_json(
        {
            "bbox": list(group.bbox),
            "region_ids": sorted(group.region_ids),
            "vertical": group.vertical,
        }
    )
    return hashlib.sha256(material).hexdigest()


_REQUIRED_EXTRAS: dict[StageName, frozenset[str]] = {
    StageName.DETECT: frozenset({"source_artifact"}),
    StageName.STYLE: frozenset(
        {"source_artifact", "style_adapter", "style_fingerprints"}
    ),
    StageName.SAFE_REGION: frozenset(
        {"safe_region_adapter", "safe_regions", "source_artifact"}
    ),
    StageName.OCR: frozenset({"ocr_adapter", "source_artifact"}),
    StageName.ORDER: frozenset(
        {"order_adapter", "ordered_region_ids", "reading_order", "source_artifact"}
    ),
    StageName.TRANSLATE: frozenset(
        {"mapping_snapshots", "provider_response_artifacts", "source_artifact"}
    ),
    StageName.LAYOUT: frozenset(
        {"layout_plan_artifact", "mapping_snapshots", "safe_regions", "source_artifact"}
    ),
    StageName.INPAINT_RENDER: frozenset(
        {
            "inpainted_image_sha256",
            "mapping_snapshots",
            "rendered_image_sha256",
            "source_artifact",
        }
    ),
}


def _validate_stage_extras(stage: StageName, extras: Mapping[str, Any]) -> None:
    required = _REQUIRED_EXTRAS.get(stage)
    if required is None:
        raise ValueError(f"stage {stage.value} does not produce pipeline state")
    missing = required - set(extras)
    if missing:
        raise ValueError(
            f"{stage.value} state is missing required extras: {sorted(missing)}"
        )


def encode_pipeline_state(
    detection: DetectionResult,
    *,
    producer_stage: StageName,
    extras: Mapping[str, Any] | None = None,
) -> StageOutputs:
    """Encode mutable legacy objects without putting masks in JSON."""

    mask_payloads: dict[str, bytes] = {}
    material_extras = dict(extras or {})
    _validate_stage_extras(producer_stage, material_extras)
    payload = {
        "schema_version": STATE_SCHEMA,
        "producer_stage": producer_stage.value,
        "detection": {
            "groups": [_group_payload(group, mask_payloads) for group in detection.groups],
            "issues": [
                {"code": issue.code, "details": issue.details, "message": issue.message}
                for issue in detection.issues
            ],
            "mask": _mask_reference(detection.mask, mask_payloads),
            "raw_mask": _mask_reference(detection.raw_mask, mask_payloads),
            "regions_post": [
                _region_payload(region, mask_payloads) for region in detection.regions_post
            ],
            "regions_raw": [
                _region_payload(region, mask_payloads) for region in detection.regions_raw
            ],
        },
        "extras": material_extras,
    }
    artifacts = [ArtifactPayload(_canonical_json(payload), STATE_MEDIA_TYPE, "state")]
    artifacts.extend(
        ArtifactPayload(data, MASK_MEDIA_TYPE, f"mask:{sha256}")
        for sha256, data in sorted(mask_payloads.items())
    )
    return StageOutputs(tuple(artifacts))


def _read_mask(
    sha256: str | None,
    *,
    allowed: set[str],
    read_bytes: Callable[[str], bytes],
) -> np.ndarray | None:
    if sha256 is None:
        return None
    if sha256 not in allowed:
        raise ValueError(f"stage state references undeclared mask artifact: {sha256}")
    payload = read_bytes(sha256)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"stage mask artifact hash mismatch: {sha256}")
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.size == 0:
        raise ValueError(f"stage mask artifact is not a decodable PNG: {sha256}")
    return mask


def _region_from_payload(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    read_bytes: Callable[[str], bytes],
) -> TextRegion:
    mask = _read_mask(payload.get("local_mask"), allowed=allowed, read_bytes=read_bytes)
    mask_source_payload = payload.get("mask_source")
    mask_source = (
        _mask_source_from_payload(mask_source_payload)
        if isinstance(mask_source_payload, Mapping)
        else None
    )
    region = TextRegion(
        id=str(payload["id"]),
        x=int(payload["x"]),
        y=int(payload["y"]),
        w=int(payload["w"]),
        h=int(payload["h"]),
        vertical=bool(payload["vertical"]),
        confidence=float(payload["confidence"]),
        source=str(payload["source"]),
        raw_index=int(payload["raw_index"]),
        detection_input_size=int(payload["detection_input_size"]),
        font_size_hint=float(payload["font_size_hint"]),
        page_bbox=(
            tuple(float(value) for value in payload["page_bbox"])
            if payload.get("page_bbox") is not None
            else None
        ),
        line_polygons=tuple(
            tuple((float(point[0]), float(point[1])) for point in polygon)
            for polygon in payload.get("line_polygons", ())
        ),
        angle_degrees=float(payload.get("angle_degrees", 0.0)),
        mask_source=mask_source,
        mask_bbox=(
            tuple(int(value) for value in payload["mask_bbox"])
            if payload.get("mask_bbox") is not None
            else None
        ),
        local_mask=mask,
        group_id=str(payload["group_id"]) if payload.get("group_id") is not None else None,
        candidate_duplicate=bool(payload["candidate_duplicate"]),
    )
    if mask is not None and mask.shape[:2] != (region.h, region.w):
        raise ValueError(f"region {region.id} mask dimensions do not match its bbox")
    return region


def _mask_source_from_payload(payload: Mapping[str, Any]) -> MaskSource:
    affine = tuple(float(value) for value in payload["page_to_local_affine"])
    if len(affine) != 6:
        raise ValueError("mask source affine transform must contain six values")
    return MaskSource(
        detector_pass=int(payload["detector_pass"]),
        detection_input_size=int(payload["detection_input_size"]),
        raw_index=int(payload["raw_index"]),
        source=str(payload["source"]),
        source_region_id=(
            str(payload["source_region_id"])
            if payload.get("source_region_id") is not None
            else None
        ),
        page_to_local_affine=affine,
    )


def _group_from_payload(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    read_bytes: Callable[[str], bytes],
) -> TextGroup:
    bbox = tuple(int(value) for value in payload["bbox"])
    if len(bbox) != 4:
        raise ValueError("group bbox must contain four integers")
    mask = _read_mask(payload.get("mask"), allowed=allowed, read_bytes=read_bytes)
    group = TextGroup(
        id=str(payload["id"]),
        region_ids=[str(value) for value in payload["region_ids"]],
        bbox=bbox,
        geometry_bbox=(
            tuple(float(value) for value in payload["geometry_bbox"])
            if payload.get("geometry_bbox") is not None
            else None
        ),
        vertical=bool(payload["vertical"]),
        ocr_text=str(payload["ocr_text"]),
        ocr_text_norm=str(payload["ocr_text_norm"]),
        ocr_confidence=float(payload["ocr_confidence"]),
        ocr_source=str(payload["ocr_source"]),
        ocr_candidates=list(payload["ocr_candidates"]),
        translation=str(payload["translation"]),
        translation_valid=bool(payload["translation_valid"]),
        status=str(payload["status"]),
        skip_reason=str(payload["skip_reason"]),
        duplicate_of=(
            str(payload["duplicate_of"]) if payload.get("duplicate_of") is not None else None
        ),
        sort_key=tuple(float(value) for value in payload["sort_key"]),
        layout_bbox=(
            tuple(int(value) for value in payload["layout_bbox"])
            if payload.get("layout_bbox") is not None
            else None
        ),
        rendered_font_size=int(payload["rendered_font_size"]),
        rendered_direction=str(payload["rendered_direction"]),
        layout_mode=str(payload["layout_mode"]),
        layout_info=dict(payload["layout_info"]),
        mapping_region_key=str(payload["mapping_region_key"]),
        mapping_chain=dict(payload["mapping_chain"]),
        mask=mask,
        mask_sources=tuple(
            _mask_source_from_payload(item) for item in payload.get("mask_sources", ())
        ),
    )
    if payload.get("stable_group_key") != _stable_group_key(group):
        raise ValueError(f"group {group.id} stable identity key mismatch")
    if mask is not None and mask.shape[:2] != (group.h, group.w):
        raise ValueError(f"group {group.id} mask dimensions do not match its bbox")
    return group


def decode_pipeline_state(
    artifacts: Sequence[ArtifactRef],
    *,
    expected_stage: StageName,
    read_bytes: Callable[[str], bytes],
) -> PipelineStageState:
    states = [artifact for artifact in artifacts if artifact.media_type == STATE_MEDIA_TYPE]
    if len(states) != 1:
        raise ValueError("stage output must contain exactly one pipeline state artifact")
    state_ref = states[0]
    raw = read_bytes(state_ref.sha256)
    if hashlib.sha256(raw).hexdigest() != state_ref.sha256:
        raise ValueError("pipeline state artifact hash mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pipeline state artifact is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported pipeline state schema")
    if payload.get("producer_stage") != expected_stage.value:
        raise ValueError(
            "pipeline state producer mismatch: "
            f"expected {expected_stage.value}, got {payload.get('producer_stage')}"
        )
    detection_payload = payload.get("detection")
    extras = payload.get("extras")
    if not isinstance(detection_payload, dict) or not isinstance(extras, dict):
        raise TypeError("pipeline state must contain detection and extras objects")
    _validate_stage_extras(expected_stage, extras)
    allowed = {
        artifact.sha256 for artifact in artifacts if artifact.media_type == MASK_MEDIA_TYPE
    }
    regions_raw = [
        _region_from_payload(item, allowed=allowed, read_bytes=read_bytes)
        for item in detection_payload["regions_raw"]
    ]
    regions_post = [
        _region_from_payload(item, allowed=allowed, read_bytes=read_bytes)
        for item in detection_payload["regions_post"]
    ]
    groups = [
        _group_from_payload(item, allowed=allowed, read_bytes=read_bytes)
        for item in detection_payload["groups"]
    ]
    issues = [
        DetectorIssue(
            code=str(item["code"]),
            message=str(item["message"]),
            details=dict(item["details"]),
        )
        for item in detection_payload["issues"]
    ]
    mask = _read_mask(detection_payload.get("mask"), allowed=allowed, read_bytes=read_bytes)
    if mask is None:
        raise ValueError("pipeline detection state is missing its required aggregate mask")
    raw_mask = _read_mask(
        detection_payload.get("raw_mask"), allowed=allowed, read_bytes=read_bytes
    )
    return PipelineStageState(
        detection=DetectionResult(
            regions_raw=regions_raw,
            regions_post=regions_post,
            groups=groups,
            mask=mask,
            raw_mask=raw_mask,
            raw_blocks=[],
            issues=issues,
        ),
        extras=dict(extras),
    )
