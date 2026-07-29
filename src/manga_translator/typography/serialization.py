"""Canonical persistence for accepted RAQM layout plans."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .fonts import FontRole
from .layout import (
    AcceptedLayout,
    FontChoice,
    LayoutCandidate,
    LayoutDirection,
)
from .shaping import ShapedFontRun
from .solver import layout_plan_hash

LAYOUT_BUNDLE_SCHEMA = "raqm_layout_bundle.v1"


def _validate_accepted_layout(group_id: str, plan: AcceptedLayout) -> None:
    if plan.alpha.ndim != 2 or not np.any(plan.alpha):
        raise ValueError(f"accepted layout {group_id} must have non-empty 2D alpha")
    if not 0.995 <= plan.containment <= 1.0:
        raise ValueError(f"accepted layout {group_id} violates alpha containment")
    if plan.candidate.font_size <= 0 or plan.candidate.tracking_em > 0.2:
        raise ValueError(f"accepted layout {group_id} has an invalid candidate")
    if not plan.shaped_runs:
        raise ValueError(f"accepted layout {group_id} has no shaped runs")
    for run in plan.shaped_runs:
        if not run.text or not run.glyph_coverage:
            raise ValueError(f"accepted layout {group_id} has incomplete glyph coverage")
        if run.direction not in {"ltr", "ttb"}:
            raise ValueError(f"accepted layout {group_id} has an invalid run direction")
    expected_hash = layout_plan_hash(plan.candidate, plan.alpha, plan.shaped_runs)
    if plan.plan_hash != expected_hash:
        raise ValueError(f"accepted layout {group_id} plan hash does not match its payload")


def _run_payload(run: ShapedFontRun) -> dict[str, Any]:
    return {
        "text": run.text,
        "font_sha256": run.font_sha256,
        "font_path": str(Path(run.font_path).resolve()),
        "glyph_coverage": list(run.glyph_coverage),
        "direction": run.direction,
        "language": run.language,
        "features": list(run.features),
        "bbox": list(run.bbox),
        "advance": run.advance,
        "anchor": list(run.anchor),
    }


def _run_from_payload(payload: dict[str, Any]) -> ShapedFontRun:
    return ShapedFontRun(
        text=str(payload["text"]),
        font_sha256=str(payload["font_sha256"]),
        font_path=str(Path(payload["font_path"]).resolve()),
        glyph_coverage=tuple(int(value) for value in payload["glyph_coverage"]),
        direction=str(payload["direction"]),
        language=str(payload["language"]),
        features=tuple(str(value) for value in payload["features"]),
        bbox=tuple(float(value) for value in payload["bbox"]),
        advance=float(payload["advance"]),
        anchor=tuple(float(value) for value in payload["anchor"]),
    )


def encode_layout_bundle(plans: dict[str, AcceptedLayout]) -> bytes:
    payload: dict[str, Any] = {"schema_version": LAYOUT_BUNDLE_SCHEMA, "plans": {}}
    for group_id, plan in sorted(plans.items()):
        _validate_accepted_layout(group_id, plan)
        encoded, alpha = cv2.imencode(".png", plan.alpha)
        if not encoded:
            raise ValueError(f"layout alpha for {group_id} could not be encoded")
        candidate = plan.candidate
        payload["plans"][group_id] = {
            "candidate": {
                "font": {
                    "role": candidate.font.role.value,
                    "weight": candidate.font.weight,
                },
                "font_size": candidate.font_size,
                "direction": candidate.direction.value,
                "chunks": list(candidate.chunks),
                "break_indices": list(candidate.break_indices),
                "line_gap_em": candidate.line_gap_em,
                "tracking_em": candidate.tracking_em,
                "anchor": list(candidate.anchor),
                "rotation_degrees": candidate.rotation_degrees,
            },
            "alpha_png": base64.b64encode(alpha.tobytes()).decode("ascii"),
            "containment": plan.containment,
            "score": plan.score,
            "plan_hash": plan.plan_hash,
            "shaped_runs": [_run_payload(run) for run in plan.shaped_runs],
            "warnings": list(plan.warnings),
        }
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


def decode_layout_bundle(raw: bytes) -> dict[str, AcceptedLayout]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("layout plan artifact is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != LAYOUT_BUNDLE_SCHEMA:
        raise ValueError("unsupported layout plan artifact schema")
    plans = payload.get("plans")
    if not isinstance(plans, dict):
        raise TypeError("layout plan artifact plans must be an object")
    decoded: dict[str, AcceptedLayout] = {}
    for group_id, value in plans.items():
        if not isinstance(value, dict) or not isinstance(value.get("candidate"), dict):
            raise TypeError("accepted layout payload must be an object")
        candidate_payload = value["candidate"]
        font_payload = candidate_payload.get("font")
        if not isinstance(font_payload, dict):
            raise TypeError("accepted layout font payload must be an object")
        alpha_raw = base64.b64decode(value["alpha_png"], validate=True)
        alpha = cv2.imdecode(np.frombuffer(alpha_raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if alpha is None or not np.any(alpha):
            raise ValueError("accepted layout alpha must be a non-empty PNG")
        candidate = LayoutCandidate(
            font=FontChoice(
                FontRole(str(font_payload["role"])), int(font_payload["weight"])
            ),
            font_size=int(candidate_payload["font_size"]),
            direction=LayoutDirection(str(candidate_payload["direction"])),
            chunks=tuple(str(item) for item in candidate_payload["chunks"]),
            break_indices=tuple(int(item) for item in candidate_payload["break_indices"]),
            line_gap_em=float(candidate_payload["line_gap_em"]),
            tracking_em=float(candidate_payload["tracking_em"]),
            anchor=tuple(float(item) for item in candidate_payload["anchor"]),
            rotation_degrees=float(candidate_payload["rotation_degrees"]),
        )
        plan = AcceptedLayout(
            candidate=candidate,
            alpha=alpha,
            containment=float(value["containment"]),
            score=float(value["score"]),
            plan_hash=str(value["plan_hash"]),
            shaped_runs=tuple(_run_from_payload(item) for item in value["shaped_runs"]),
            warnings=tuple(str(item) for item in value["warnings"]),
        )
        _validate_accepted_layout(str(group_id), plan)
        decoded[str(group_id)] = plan
    return decoded
