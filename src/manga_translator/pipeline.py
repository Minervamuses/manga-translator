"""主流水線：detect → mask fallback → OCR ensemble → fuzzy dedup → translate → safe render。"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .artifacts import dump_debug_artifacts, dump_page_document
from .config import AppConfig, PostprocessConfig
from .contracts.mapping import (
    MappingContractError,
    MappingIssue,
    RawResponseRef,
    RequestItem,
    RequestMap,
    ResponseItem,
    ValidatedTranslationBatch,
    bind_validated_responses,
    bind_validated_values,
    build_request_map,
    mapping_chain_template,
)
from .detector import DetectionResult, TextGroup, TextRegion, detect_text_regions
from .domain.issues import Issue, IssueCode, IssueSeverity, StageName, StageStatus
from .domain.models import (
    ArtifactRef,
    BoundingBox,
    EntityRecord,
    OCRCandidate,
    OCRRecord,
    PageDocument,
    RegionRevision,
    SourcePage,
    StageRecord,
    TranslationRecord,
)
from .domain.reconcile import RegionObservation, reconcile_regions
from .geometry import center_distance, containment_ratio, iom, merge_bbox
from .image_io import (
    ImageEncodeError,
    ImageWriteError,
    read_image,
    write_image,
    write_image_or_raise,
)
from .inpainter import inpaint_regions
from .ocr import (
    OCRInitializationError,
    assess_ocr_result,
    initialize_ocr_model,
    normalize_ocr_text,
    ocr_group_detailed,
)
from .profiling import profile_page, profile_span, set_page_profile_metrics
from .result import (
    BatchResult,
    GroupMappingSnapshot,
    PageResult,
    ResultIssue,
    derive_batch_status,
)
from .stages.adapters import (
    build_pipeline_stage_specs,
    glossary_fingerprint,
    stage_config,
)
from .stages.base import (
    ArtifactPayload,
    StageContext,
    StageFunction,
    StageInputs,
    StageOutputs,
)
from .stages.runner import StageOutcome, StageRunner
from .stages.state import PipelineStageState, decode_pipeline_state, encode_pipeline_state
from .storage import ArtifactStore, JobStore
from .translator import (
    load_glossary,
    sanitize_translation_text,
    translate_batch_mapped,
    translate_page_mapped,
    translate_with_context_mapped,
    validate_translation,
)
from .typesetter import (
    TextLayoutPlan,
    layout_plan_block_bbox,
    plan_text_layout,
    render_text_into_group,
)

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRANSLATION_BUNDLE_SCHEMA_V1 = "translation_response_bundle.v1"
TRANSLATION_BUNDLE_SCHEMA = "translation_response_bundle.v2"


class TranslationBundleReplayError(RuntimeError):
    """A provider response was durable, but its semantic outcome was rejection."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        mapping_issues: tuple[MappingIssue, ...],
        raw_artifacts: dict[str, tuple[RawResponseRef, bytes]],
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.mapping_issues = mapping_issues
        self.raw_artifacts = raw_artifacts


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
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


def _raw_response_payload(reference: RawResponseRef, artifact_root: Path) -> bytes:
    if reference.relative_path is None:
        raise ValueError("provider response reference has no durable relative path")
    relative = Path(reference.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("provider response artifact path must remain below its artifact root")
    root = artifact_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("provider response artifact escaped its artifact root") from error
    payload = path.read_bytes()
    if len(payload) != reference.size_bytes:
        raise ValueError("provider response artifact size does not match its reference")
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("provider response artifact hash does not match its reference")
    return payload


def _request_bundle_payload(request: RequestMap) -> dict[str, object]:
    return {
        "items": [
            {
                "item_id": item.item_id,
                "region_key": item.region_key,
                "source_sha256": item.source_sha256,
                "source_text": item.source_text,
            }
            for item in request.items
        ],
        "page_id": request.page_id,
        "request_id": request.request_id,
    }


def _raw_bundle_entries(
    references: tuple[RawResponseRef, ...],
    *,
    artifact_root: Path,
) -> list[dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for reference in references:
        raw = _raw_response_payload(reference, artifact_root)
        existing = entries.get(reference.sha256)
        entry = {
            **reference.to_dict(),
            "payload_base64": base64.b64encode(raw).decode("ascii"),
        }
        if existing is not None:
            if (
                existing["media_type"] != entry["media_type"]
                or existing["size_bytes"] != entry["size_bytes"]
                or existing["payload_base64"] != entry["payload_base64"]
            ):
                raise ValueError("conflicting provider responses share one SHA-256 identity")
            continue
        entries[reference.sha256] = entry
    return [entries[key] for key in sorted(entries)]


def _serialize_translation_bundle(
    batch: ValidatedTranslationBatch,
    *,
    artifact_root: Path,
) -> bytes:
    if any(response.raw_response_ref is None for response in batch.responses):
        raise ValueError("durable translation responses require exact provider artifacts")
    raw_references = tuple(
        response.raw_response_ref
        for response in batch.responses
        if response.raw_response_ref is not None
    )
    responses: list[dict[str, object]] = []
    for response in batch.responses:
        raw_reference = response.raw_response_ref
        responses.append(
            {
                "item_id": response.item_id,
                "raw_response_sha256": (
                    raw_reference.sha256 if raw_reference is not None else None
                ),
                "response_index": response.response_index,
                "source_sha256": response.source_sha256,
                "translation": response.translation,
            }
        )
    return _canonical_json_bytes(
        {
            "outcome": {"status": "succeeded"},
            "raw_responses": _raw_bundle_entries(
                raw_references,
                artifact_root=artifact_root,
            ),
            "request": _request_bundle_payload(batch.request),
            "responses": responses,
            "schema_version": TRANSLATION_BUNDLE_SCHEMA,
        }
    )


def _serialize_translation_failure_bundle(
    request: RequestMap,
    error: Exception,
    *,
    artifact_root: Path,
) -> bytes:
    raw_references = tuple(getattr(error, "raw_response_refs", ()))
    if not raw_references:
        raise ValueError("cannot persist a provider failure without raw response bytes")
    mapping_issues = (
        tuple(error.issues) if isinstance(error, MappingContractError) else ()
    )
    return _canonical_json_bytes(
        {
            "outcome": {
                "error_type": type(error).__name__,
                "mapping_issues": [
                    {"code": issue.code, "details": dict(issue.details)}
                    for issue in mapping_issues
                ],
                "message": str(error),
                "status": "failed",
            },
            "raw_responses": _raw_bundle_entries(
                raw_references,
                artifact_root=artifact_root,
            ),
            "request": _request_bundle_payload(request),
            "responses": [],
            "schema_version": TRANSLATION_BUNDLE_SCHEMA,
        }
    )


def _request_from_bundle(payload: object) -> RequestMap:
    if not isinstance(payload, dict):
        raise TypeError("translation response bundle request must be an object")
    item_payloads = payload.get("items")
    if not isinstance(item_payloads, list):
        raise TypeError("translation response bundle request items must be an array")
    items: list[RequestItem] = []
    for item in item_payloads:
        if not isinstance(item, dict):
            raise TypeError("translation response bundle request items must be objects")
        items.append(
            RequestItem(
                item_id=str(item["item_id"]),
                region_key=str(item["region_key"]),
                source_text=str(item["source_text"]),
                source_sha256=str(item["source_sha256"]),
            )
        )
    return RequestMap(
        request_id=str(payload["request_id"]),
        page_id=str(payload["page_id"]),
        items=tuple(items),
    )


def _decode_raw_bundle_entries(
    payloads: object,
) -> dict[str, tuple[RawResponseRef, bytes]]:
    if not isinstance(payloads, list):
        raise TypeError("translation raw responses must be an array")
    raw_artifacts: dict[str, tuple[RawResponseRef, bytes]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError("translation raw response entries must be objects")
        try:
            content = base64.b64decode(str(payload["payload_base64"]), validate=True)
        except (KeyError, ValueError) as error:
            raise ValueError("translation raw response payload is not valid base64") from error
        reference = RawResponseRef(
            sha256=str(payload["sha256"]),
            media_type=str(payload["media_type"]),
            size_bytes=int(payload["size_bytes"]),
            relative_path=(
                str(payload["relative_path"])
                if payload.get("relative_path") is not None
                else None
            ),
        )
        if len(content) != reference.size_bytes:
            raise ValueError("bundled provider response size mismatch")
        if hashlib.sha256(content).hexdigest() != reference.sha256:
            raise ValueError("bundled provider response hash mismatch")
        existing = raw_artifacts.get(reference.sha256)
        if existing is not None and (
            existing[0].media_type != reference.media_type
            or existing[0].size_bytes != reference.size_bytes
            or existing[1] != content
        ):
            raise ValueError("conflicting provider responses share one SHA-256 identity")
        raw_artifacts[reference.sha256] = (reference, content)
    return raw_artifacts


def _deserialize_translation_bundle_v1(
    request: RequestMap,
    response_payloads: list[object],
) -> tuple[ValidatedTranslationBatch, dict[str, tuple[RawResponseRef, bytes]]]:
    raw_artifacts: dict[str, tuple[RawResponseRef, bytes]] = {}
    responses: list[ResponseItem] = []
    for item in response_payloads:
        if not isinstance(item, dict):
            raise TypeError("translation response bundle items must be objects")
        raw_payload = item.get("raw_response")
        reference = None
        if raw_payload is not None:
            decoded = _decode_raw_bundle_entries([raw_payload])
            reference, _content = next(iter(decoded.values()))
            existing = raw_artifacts.get(reference.sha256)
            if existing is not None and existing != decoded[reference.sha256]:
                raise ValueError("conflicting provider responses share one SHA-256 identity")
            raw_artifacts.update(decoded)
        responses.append(
            ResponseItem(
                item_id=str(item["item_id"]),
                source_sha256=str(item["source_sha256"]),
                translation=str(item["translation"]),
                response_index=int(item["response_index"]),
                raw_response_ref=reference,
            )
        )
    return bind_validated_responses(request, responses), raw_artifacts


def _deserialize_translation_bundle(
    raw: bytes,
) -> tuple[ValidatedTranslationBatch, dict[str, tuple[RawResponseRef, bytes]]]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("translation response bundle is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        TRANSLATION_BUNDLE_SCHEMA_V1,
        TRANSLATION_BUNDLE_SCHEMA,
    }:
        raise ValueError("unsupported translation response bundle schema")
    request = _request_from_bundle(payload.get("request"))
    response_payloads = payload.get("responses")
    if not isinstance(response_payloads, list):
        raise TypeError("translation response bundle responses must be an array")
    if payload["schema_version"] == TRANSLATION_BUNDLE_SCHEMA_V1:
        return _deserialize_translation_bundle_v1(request, response_payloads)

    raw_artifacts = _decode_raw_bundle_entries(payload.get("raw_responses"))
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        raise TypeError("translation response bundle outcome must be an object")
    status = outcome.get("status")
    if status == "failed":
        issue_payloads = outcome.get("mapping_issues", [])
        if not isinstance(issue_payloads, list):
            raise TypeError("translation failure mapping issues must be an array")
        issues: list[MappingIssue] = []
        for item in issue_payloads:
            if not isinstance(item, dict) or not isinstance(item.get("details"), dict):
                raise TypeError("translation failure mapping issues must be objects")
            issues.append(MappingIssue(str(item["code"]), dict(item["details"])))
        raise TranslationBundleReplayError(
            str(outcome.get("message", "provider response was rejected")),
            error_type=str(outcome.get("error_type", "ProviderResponseError")),
            mapping_issues=tuple(issues),
            raw_artifacts=raw_artifacts,
        )
    if status != "succeeded":
        raise ValueError("translation response bundle has an unsupported outcome")

    responses: list[ResponseItem] = []
    for item in response_payloads:
        if not isinstance(item, dict):
            raise TypeError("translation response bundle items must be objects")
        raw_sha256 = item.get("raw_response_sha256")
        reference = (
            raw_artifacts[str(raw_sha256)][0] if raw_sha256 is not None else None
        )
        responses.append(
            ResponseItem(
                item_id=str(item["item_id"]),
                source_sha256=str(item["source_sha256"]),
                translation=str(item["translation"]),
                response_index=int(item["response_index"]),
                raw_response_ref=reference,
            )
        )
    return bind_validated_responses(request, responses), raw_artifacts


def _natural_sort_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=_natural_sort_key)


def _group_area(group: TextGroup) -> int:
    return max(0, group.w) * max(0, group.h)


def _group_area_ratio(a: TextGroup, b: TextGroup) -> float:
    small = min(_group_area(a), _group_area(b))
    large = max(_group_area(a), _group_area(b), 1)
    return small / large


def _text_similarity(a: str, b: str) -> float:
    left = normalize_ocr_text(a, weak=True)
    right = normalize_ocr_text(b, weak=True)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _axis_overlap_ratio(a: TextGroup, b: TextGroup) -> float:
    """同一行／同一列的重疊程度；用來限制相鄰框去重。"""
    if a.vertical and b.vertical:
        overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return overlap / max(1, min(a.h, b.h))
    if not a.vertical and not b.vertical:
        overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        return overlap / max(1, min(a.w, b.w))
    return 0.0


def _bbox_gap_ratio(a: TextGroup, b: TextGroup) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    dx = max(0, max(b.x - ax2, a.x - bx2))
    dy = max(0, max(b.y - ay2, a.y - by2))
    gap = max(dx, dy)
    return gap / max(1, min(max(a.w, a.h), max(b.w, b.h)))


def _local_group_mask(group: TextGroup) -> np.ndarray | None:
    """回傳與 group.bbox 對齊的 mask，並相容舊版整頁 mask。"""
    if group.mask is None or group.mask.size == 0:
        return None
    if group.mask.shape[:2] == (group.h, group.w):
        return group.mask
    if group.mask.shape[0] >= group.y + group.h and group.mask.shape[1] >= group.x + group.w:
        return group.mask[group.y : group.y + group.h, group.x : group.x + group.w]
    return cv2.resize(group.mask, (group.w, group.h), interpolation=cv2.INTER_NEAREST)


def _paste_group_mask(
    canvas: np.ndarray,
    group: TextGroup,
    canvas_bbox: tuple[int, int, int, int],
) -> None:
    local = _local_group_mask(group)
    if local is None:
        return
    canvas_x, canvas_y, _canvas_w, _canvas_h = canvas_bbox
    x1 = group.x - canvas_x
    y1 = group.y - canvas_y
    x2 = x1 + group.w
    y2 = y1 + group.h
    if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
        return
    canvas[y1:y2, x1:x2] = cv2.bitwise_or(canvas[y1:y2, x1:x2], local)


def _group_mask_containment(inner: TextGroup, outer: TextGroup) -> float:
    """Return the fraction of ``inner`` text pixels also covered by ``outer``.

    Multi-scale detection often yields a whole-sentence mask plus one mask per
    vertical column.  Their mask IoU is small because the whole sentence is much
    larger, but nearly every pixel in the column is still contained by the outer
    mask.  IoU alone therefore misses exactly the overlap that later creates
    doubled subtitles.
    """
    inner_mask = _local_group_mask(inner)
    outer_mask = _local_group_mask(outer)
    if inner_mask is None or outer_mask is None:
        return 0.0

    inner_count = int(np.count_nonzero(inner_mask))
    if inner_count <= 0:
        return 0.0

    overlap_x1 = max(inner.x, outer.x)
    overlap_y1 = max(inner.y, outer.y)
    overlap_x2 = min(inner.x + inner.w, outer.x + outer.w)
    overlap_y2 = min(inner.y + inner.h, outer.y + outer.h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    inner_crop = inner_mask[
        overlap_y1 - inner.y : overlap_y2 - inner.y,
        overlap_x1 - inner.x : overlap_x2 - inner.x,
    ]
    outer_crop = outer_mask[
        overlap_y1 - outer.y : overlap_y2 - outer.y,
        overlap_x1 - outer.x : overlap_x2 - outer.x,
    ]
    if inner_crop.shape != outer_crop.shape or inner_crop.size == 0:
        return 0.0

    intersection = int(np.count_nonzero((inner_crop > 0) & (outer_crop > 0)))
    return intersection / inner_count


def _text_coverage(a: str, b: str) -> float:
    """How much of the shorter normalized string is explained by the longer."""
    left = normalize_ocr_text(a, weak=True)
    right = normalize_ocr_text(b, weak=True)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    shorter, longer = sorted((left, right), key=len)
    blocks = SequenceMatcher(None, shorter, longer, autojunk=False).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    return matched / max(1, len(shorter))


def _group_mask_iou(a: TextGroup, b: TextGroup) -> float:
    local_a = _local_group_mask(a)
    local_b = _local_group_mask(b)
    if local_a is None or local_b is None:
        return 0.0

    count_a = int(np.count_nonzero(local_a))
    count_b = int(np.count_nonzero(local_b))
    if count_a <= 0 or count_b <= 0:
        return 0.0

    overlap_x1 = max(a.x, b.x)
    overlap_y1 = max(a.y, b.y)
    overlap_x2 = min(a.x + a.w, b.x + b.w)
    overlap_y2 = min(a.y + a.h, b.y + b.h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    a_crop = local_a[
        overlap_y1 - a.y : overlap_y2 - a.y,
        overlap_x1 - a.x : overlap_x2 - a.x,
    ]
    b_crop = local_b[
        overlap_y1 - b.y : overlap_y2 - b.y,
        overlap_x1 - b.x : overlap_x2 - b.x,
    ]
    if a_crop.shape != b_crop.shape or a_crop.size == 0:
        return 0.0

    intersection = int(np.count_nonzero((a_crop > 0) & (b_crop > 0)))
    union = count_a + count_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _are_duplicate_groups(a: TextGroup, b: TextGroup, cfg: PostprocessConfig) -> bool:
    iom_score = iom(a, b)
    high_overlap = iom_score >= cfg.same_text_iom_thresh
    containment_ab = containment_ratio(a, b)
    containment_ba = containment_ratio(b, a)
    bbox_containment = max(containment_ab, containment_ba)
    contained = bbox_containment >= cfg.containment_ratio_thresh
    area_ratio = _group_area_ratio(a, b)
    max_dim = max(a.w, a.h, b.w, b.h, 1)
    center_ratio = center_distance(a, b) / max_dim
    mask_iou = _group_mask_iou(a, b)
    mask_containment_ab = _group_mask_containment(a, b)
    mask_containment_ba = _group_mask_containment(b, a)
    mask_containment = max(mask_containment_ab, mask_containment_ba)

    nested_geometry = (
        a.vertical == b.vertical
        and bbox_containment >= cfg.nested_fragment_containment
        and (mask_containment >= cfg.render_collision_mask_containment or center_ratio <= 0.82)
    )
    strong_geometry = (
        mask_iou >= 0.42
        or mask_containment >= cfg.render_collision_mask_containment
        or (high_overlap and area_ratio >= 0.30)
        or (contained and area_ratio >= 0.45 and center_ratio <= 0.70)
    )

    a_text = a.ocr_text_norm
    b_text = b.ocr_text_norm
    if strong_geometry and (not a_text or not b_text):
        # Empty OCR candidates are merged only when their actual text pixels overlap
        # strongly.  A giant empty panel bbox must not swallow real speech bubbles.
        return mask_iou >= 0.55 or (mask_containment >= 0.88 and area_ratio >= 0.18)
    if not a_text or not b_text:
        return False

    similarity = _text_similarity(a_text, b_text)
    coverage = _text_coverage(a_text, b_text)
    exact_or_substring = (
        a_text == b_text
        or (
            cfg.substring_match_enabled
            and (a_text in b_text or b_text in a_text)
            and min(len(a_text), len(b_text)) >= 2
        )
    )

    # Whole-sentence box + one-column fragment: the shorter OCR can have one or
    # two wrong glyphs, so ordinary similarity is often too low.  Geometry plus
    # coverage of the shorter text is the reliable signal.
    if nested_geometry and coverage >= cfg.nested_fragment_text_coverage:
        return True

    if strong_geometry:
        fuzzy_threshold = max(0.62, cfg.fuzzy_text_similarity_thresh - 0.12)
        return exact_or_substring or coverage >= 0.78 or similarity >= fuzzy_threshold

    # Non-overlapping boxes must be very close, aligned and textually near-identical.
    if a.vertical != b.vertical:
        return False
    near = _bbox_gap_ratio(a, b) <= cfg.duplicate_near_gap_ratio
    aligned = _axis_overlap_ratio(a, b) >= 0.58
    centers_close = center_ratio <= min(cfg.group_center_dist_ratio, 0.85)
    size_similar = area_ratio >= 0.55
    return near and aligned and centers_close and (
        size_similar
        and _bbox_gap_ratio(a, b) <= min(cfg.duplicate_near_gap_ratio, 0.08)
        and (exact_or_substring or similarity >= max(0.90, cfg.fuzzy_text_similarity_thresh))
    )


def _merge_group_masks(
    a: TextGroup,
    b: TextGroup,
    merged_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    if a.mask is None and b.mask is None:
        return None
    _x, _y, w, h = merged_bbox
    merged = np.zeros((h, w), dtype=np.uint8)
    _paste_group_mask(merged, a, merged_bbox)
    _paste_group_mask(merged, b, merged_bbox)
    return merged


def _best_ocr_group(a: TextGroup, b: TextGroup) -> TextGroup:
    a_norm = a.ocr_text_norm
    b_norm = b.ocr_text_norm
    if a_norm and b_norm:
        if a_norm in b_norm and len(b_norm) > len(a_norm):
            return b
        if b_norm in a_norm and len(a_norm) > len(b_norm):
            return a

        # In a nested whole-box/column pair, favor the more complete OCR unless
        # its quality is clearly worse.  Picking a tiny fragment just because its
        # confidence is 0.02 higher is what used to create incomplete/repeated text.
        if _text_coverage(a_norm, b_norm) >= 0.62 and abs(len(a_norm) - len(b_norm)) >= 2:
            longer, shorter = (a, b) if len(a_norm) > len(b_norm) else (b, a)
            if longer.ocr_confidence >= shorter.ocr_confidence - 0.16:
                return longer

    return max(
        (a, b),
        key=lambda group: (
            float(group.ocr_confidence) + min(0.12, len(group.ocr_text_norm) * 0.006),
            len(group.ocr_text_norm),
            _group_area(group),
        ),
    )


def _best_translation_group(a: TextGroup, b: TextGroup) -> TextGroup:
    return max(
        (a, b),
        key=lambda group: (
            bool(group.translation_valid and group.translation.strip()),
            len(group.translation),
            group.ocr_confidence,
        ),
    )


def _merge_group_objects(
    a: TextGroup,
    b: TextGroup,
    *,
    identity_source: TextGroup | None = None,
) -> TextGroup:
    best = _best_ocr_group(a, b)
    merged_bbox = merge_bbox([a, b])
    translation_best = _best_translation_group(a, b)
    identity_source = identity_source or a
    merged_candidates: list[dict[str, object]] = []
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in a.ocr_candidates + b.ocr_candidates:
        key = (str(candidate.get("source", "")), str(candidate.get("normalized", "")))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        merged_candidates.append(candidate)

    return TextGroup(
        id=identity_source.id,
        region_ids=list(dict.fromkeys(a.region_ids + b.region_ids)),
        bbox=merged_bbox,
        vertical=a.vertical if _group_area(a) >= _group_area(b) else b.vertical,
        ocr_text=best.ocr_text,
        ocr_text_norm=best.ocr_text_norm,
        ocr_confidence=best.ocr_confidence,
        ocr_source=best.ocr_source,
        ocr_candidates=merged_candidates,
        translation=translation_best.translation,
        translation_valid=translation_best.translation_valid,
        status=translation_best.status if translation_best.translation_valid else best.status,
        skip_reason="" if translation_best.translation_valid else best.skip_reason,
        sort_key=min(a.sort_key, b.sort_key),
        mapping_region_key=translation_best.mapping_region_key,
        mapping_chain=dict(translation_best.mapping_chain),
        mask=_merge_group_masks(a, b, merged_bbox),
    )


def _merge_duplicate_groups(
    groups: list[TextGroup],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_ocr_dedup or len(groups) < 2:
        return groups

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a_index: int, b_index: int) -> None:
        root_a, root_b = find(a_index), find(b_index)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _are_duplicate_groups(groups[i], groups[j], cfg):
                union(i, j)

    components: dict[int, list[TextGroup]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)

    merged: list[TextGroup] = []
    for component in components.values():
        current = component[0]
        for duplicate in component[1:]:
            duplicate.duplicate_of = current.id
            current = _merge_group_objects(current, duplicate)
        merged.append(current)
    return merged


def _are_translation_duplicates(
    a: TextGroup,
    b: TextGroup,
    cfg: PostprocessConfig,
) -> bool:
    """Second guard for OCR variants that translate to the same sentence."""
    if not (a.translation_valid and b.translation_valid):
        return False
    left = sanitize_translation_text(a.translation, source=a.ocr_text)
    right = sanitize_translation_text(b.translation, source=b.ocr_text)
    if not left or not right:
        return False

    translation_similarity = _text_similarity(left, right)
    if translation_similarity < 0.94:
        return False

    area_ratio = _group_area_ratio(a, b)
    bbox_containment = max(containment_ratio(a, b), containment_ratio(b, a))
    mask_containment = max(
        _group_mask_containment(a, b),
        _group_mask_containment(b, a),
    )
    strong_overlap = (
        _group_mask_iou(a, b) >= 0.42
        or mask_containment >= cfg.render_collision_mask_containment
        or (iom(a, b) >= max(0.65, cfg.same_text_iom_thresh) and area_ratio >= 0.45)
        or (bbox_containment >= cfg.containment_ratio_thresh and area_ratio >= 0.60)
    )
    if not strong_overlap:
        return False

    # 「嗯」「啊」等短句很常在相鄰氣泡合法重複，還要有來源文字相似訊號。
    compact = normalize_ocr_text(left, weak=True)
    return not (len(compact) <= 2 and _text_similarity(a.ocr_text_norm, b.ocr_text_norm) < 0.78)


def _merge_translation_duplicates(
    groups: list[TextGroup],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_ocr_dedup or len(groups) < 2:
        return groups

    kept: list[TextGroup] = []
    consumed = [False] * len(groups)
    for index, group in enumerate(groups):
        if consumed[index]:
            continue
        current = group
        members = [group]
        for other_index in range(index + 1, len(groups)):
            if consumed[other_index]:
                continue
            other = groups[other_index]
            if not _are_translation_duplicates(current, other, cfg):
                continue
            survivor = _best_translation_group(current, other)
            survivor_request = survivor.mapping_chain.get("request_item")
            members.append(other)
            for member in members:
                member.duplicate_of = (
                    None
                    if member.mapping_chain.get("request_item") == survivor_request
                    else survivor.id
                )
            current = _merge_group_objects(
                current,
                other,
                identity_source=survivor,
            )
            consumed[other_index] = True
        kept.append(current)
    return kept


def _group_source_rank(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> tuple[int, int, int]:
    sources = [
        regions_by_id[rid].source
        for rid in group.region_ids
        if rid in regions_by_id
    ]
    return (
        sum(source == "ctd" for source in sources),
        sum(source == "ctd_multiscale" for source in sources),
        -sum(source == "mask_fallback" for source in sources),
    )


def _render_group_score(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> tuple[object, ...]:
    local_mask = _local_group_mask(group)
    mask_pixels = int(np.count_nonzero(local_mask)) if local_mask is not None else 0
    return (
        _group_source_rank(group, regions_by_id),
        float(group.ocr_confidence),
        len(group.ocr_text_norm),
        mask_pixels,
        -_group_area(group),
    )


def _render_groups_conflict(
    a: TextGroup,
    b: TextGroup,
    cfg: PostprocessConfig,
) -> bool:
    """Prevent two translations from being drawn over the same text pixels.

    This guard intentionally does not trust OCR strings.  A whole-sentence mask
    and one nested column have low IoU and a tiny area ratio, yet nearly all pixels
    of the column are contained in the whole mask; mask containment catches it.
    """
    area_ratio = _group_area_ratio(a, b)
    mask_overlap = _group_mask_iou(a, b)
    mask_containment = max(
        _group_mask_containment(a, b),
        _group_mask_containment(b, a),
    )
    bbox_iom = iom(a, b)
    contained = max(containment_ratio(a, b), containment_ratio(b, a))
    nested_bbox = (
        a.vertical == b.vertical
        and contained >= cfg.nested_fragment_containment
        and area_ratio >= 0.08
    )
    return (
        mask_overlap >= cfg.render_collision_mask_iou
        or mask_containment >= cfg.render_collision_mask_containment
        or (bbox_iom >= cfg.render_collision_iom and area_ratio >= 0.50)
        or (contained >= cfg.render_collision_containment and area_ratio >= 0.55)
        or nested_bbox
    )


def _resolve_render_collisions(
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if not cfg.enable_render_collision_filter:
        return groups

    renderable = [
        group
        for group in groups
        if group.translation_valid and bool(group.translation.strip())
    ]
    ordered = sorted(
        renderable,
        key=lambda group: _render_group_score(group, regions_by_id),
        reverse=True,
    )
    accepted: list[TextGroup] = []
    for group in ordered:
        conflict = next(
            (
                winner
                for winner in accepted
                if _render_groups_conflict(group, winner, cfg)
            ),
            None,
        )
        if conflict is None:
            accepted.append(group)
            continue

        group.translation_valid = False
        group.translation = ""
        group.status = "render_collision_rejected"
        group.skip_reason = f"overlaps:{conflict.id}"
        group.duplicate_of = conflict.id
    return groups


def _layout_blocks_conflict(a: TextLayoutPlan, b: TextLayoutPlan) -> bool:
    """Final safety guard based on the text that will actually be drawn."""

    ax, ay, aw, ah = layout_plan_block_bbox(a)
    bx, by, bw, bh = layout_plan_block_bbox(b)
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    if overlap_w <= 0 or overlap_h <= 0:
        return False

    glyph_guard = max(2.0, min(a.font_size, b.font_size) * 0.24)
    if overlap_w < glyph_guard or overlap_h < glyph_guard:
        return False
    overlap_area = overlap_w * overlap_h
    return overlap_area / max(1, min(aw * ah, bw * bh)) >= 0.06


def _record_layout_plan(
    group: TextGroup,
    plan: TextLayoutPlan,
    *,
    mode: str,
) -> None:
    group.layout_bbox = plan.bbox
    group.rendered_font_size = int(plan.font_size)
    group.rendered_direction = plan.direction
    group.layout_mode = mode
    group.layout_info = plan.to_dict()
    group.layout_info["block_bbox"] = {
        key: value
        for key, value in zip(("x", "y", "w", "h"), layout_plan_block_bbox(plan))
    }


def _preflight_layout_plans(
    original: np.ndarray,
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    config: AppConfig,
) -> dict[str, TextLayoutPlan]:
    """Plan before erasing so a tiny/overlapping result keeps its original text.

    This is intentionally run before inpainting.  A layout that would need a font
    below the readability floor, or that would collide with a higher-quality
    subtitle, is rejected while the original Japanese pixels are still present.
    """

    renderable = [
        group
        for group in groups
        if group.translation_valid and bool(group.translation.strip())
    ]
    ordered = sorted(
        renderable,
        key=lambda group: _render_group_score(group, regions_by_id),
        reverse=True,
    )
    accepted: list[tuple[TextGroup, TextLayoutPlan]] = []
    plans: dict[str, TextLayoutPlan] = {}

    fallback_cfg = config.typesetting.model_copy(
        update={
            "adaptive_bubble_layout": False,
            "layout_padding_ratio": min(config.typesetting.layout_padding_ratio, 0.08),
        }
    )

    for group in ordered:
        plan = plan_text_layout(
            original,
            group,
            regions_by_id,
            group.translation,
            config.paths.font,
            config.typesetting,
            config.paths.font_fallback,
        )
        mode = config.typesetting.layout_mode
        if not plan.fits:
            group.translation_valid = False
            group.status = "layout_rejected"
            group.skip_reason = plan.reason or "unreadable_layout"
            _record_layout_plan(group, plan, mode=mode)
            continue

        collision = next(
            (winner for winner, winner_plan in accepted if _layout_blocks_conflict(plan, winner_plan)),
            None,
        )
        if collision is not None:
            compact = plan_text_layout(
                original,
                group,
                regions_by_id,
                group.translation,
                config.paths.font,
                fallback_cfg,
                config.paths.font_fallback,
            )
            compact_collision = (
                next(
                    (
                        winner
                        for winner, winner_plan in accepted
                        if compact.fits and _layout_blocks_conflict(compact, winner_plan)
                    ),
                    None,
                )
                if compact.fits
                else collision
            )
            if compact.fits and compact_collision is None:
                plan = compact
                mode = f"{config.typesetting.layout_mode}:no_bubble_expand"
            else:
                group.translation_valid = False
                group.status = "layout_collision_rejected"
                group.skip_reason = f"planned_text_overlaps:{collision.id}"
                _record_layout_plan(group, plan, mode=mode)
                continue

        _record_layout_plan(group, plan, mode=mode)
        accepted.append((group, plan))
        plans[group.id] = plan

    return plans


def _record_mapping_layout_plans(groups: list[TextGroup]) -> None:
    for group in groups:
        if group.layout_info and group.mapping_chain:
            group.mapping_chain["layout_plan"] = f"layout:{group.id}"


def _refresh_group_order(
    groups: list[TextGroup],
    regions_by_id: dict[str, TextRegion],
    cfg: PostprocessConfig,
) -> list[TextGroup]:
    if cfg.reading_order == "jp_vertical":
        use_vertical_order = True
        groups = sorted(
            groups,
            key=lambda group: (
                -(group.x + group.w / 2.0),
                group.y + group.h / 2.0,
            ),
        )
    else:
        vertical_ratio = sum(group.vertical for group in groups) / max(1, len(groups))
        use_vertical_order = vertical_ratio >= 0.5
        if use_vertical_order:
            groups = sorted(
                groups,
                key=lambda group: (-(group.x + group.w / 2.0), group.y + group.h / 2.0),
            )
        else:
            groups = sorted(
                groups,
                key=lambda group: (group.y + group.h / 2.0, group.x + group.w / 2.0),
            )

    for index, group in enumerate(groups):
        group.id = f"g{index:03d}"
        if use_vertical_order:
            group.sort_key = (-(group.x + group.w / 2.0), group.y + group.h / 2.0)
        else:
            group.sort_key = (group.y + group.h / 2.0, group.x + group.w / 2.0)
        for region_id in group.region_ids:
            region = regions_by_id.get(region_id)
            if region is not None:
                region.group_id = group.id
    return groups


def _build_page_translation_units(groups: list[TextGroup]) -> tuple[list[TextGroup], list[str]]:
    # 呼叫者已經完成閱讀順序排序。此處不可再自行重排，否則回傳翻譯與
    # 原 group 逐項 zip 時可能錯置到別的對話框。
    ordered = list(groups)
    return ordered, [group.ocr_text for group in ordered]


def _mapping_region_key(page_id: str, group: TextGroup) -> str:
    identity_parts = sorted(group.region_ids) or [group.id]
    material = "|".join(
        [page_id, *identity_parts, *(str(value) for value in group.bbox)]
    )
    return "group:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _build_translation_request(
    groups: list[TextGroup],
    page_id: str,
) -> tuple[list[TextGroup], list[str], RequestMap]:
    ordered, texts = _build_page_translation_units(groups)
    for group in ordered:
        if not group.mapping_region_key:
            group.mapping_region_key = _mapping_region_key(page_id, group)
    request = build_request_map(
        page_id, ((group.mapping_region_key, group.ocr_text) for group in ordered)
    )
    return ordered, texts, request


def _request_translations(
    groups: list[TextGroup],
    page_id: str,
    config: AppConfig,
    glossary: dict[str, str],
) -> ValidatedTranslationBatch:
    _ordered, texts, request = _build_translation_request(groups, page_id)
    if not texts:
        return bind_validated_values(request, [])
    item_ids = [item.item_id for item in request.items]

    if not config.postprocess.enable_group_translate:
        if config.openrouter.translation_mode == "context":
            responses = translate_with_context_mapped(
                texts,
                config.openrouter,
                glossary,
                context_size=config.openrouter.context_size,
                item_ids=item_ids,
                artifact_root=config.paths.output_dir,
            )
        else:
            responses = translate_batch_mapped(
                texts,
                config.openrouter,
                glossary,
                item_ids=item_ids,
                artifact_root=config.paths.output_dir,
            )
        return bind_validated_responses(request, responses)

    total_chars = sum(len(text) for text in texts)
    should_fallback_window = total_chars > 6000 or len(texts) > 120
    if config.openrouter.page_context_mode == "page" and not should_fallback_window:
        responses = translate_page_mapped(
            texts,
            config.openrouter,
            glossary,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    elif config.openrouter.translation_mode == "context":
        responses = translate_with_context_mapped(
            texts,
            config.openrouter,
            glossary,
            context_size=config.openrouter.context_size,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    else:
        responses = translate_batch_mapped(
            texts,
            config.openrouter,
            glossary,
            item_ids=item_ids,
            artifact_root=config.paths.output_dir,
        )
    return bind_validated_responses(request, responses)


def _prepare_translation_groups(
    groups: list[TextGroup],
    page_id: str,
    config: AppConfig,
) -> list[TextGroup]:
    translatable: list[TextGroup] = []
    for group in groups:
        if not group.mapping_region_key:
            group.mapping_region_key = _mapping_region_key(page_id, group)
        group.mapping_chain = mapping_chain_template(
            region_key=group.mapping_region_key,
            ocr_record=f"ocr:{group.mapping_region_key}",
        )
        if group.status in {"ocr_rejected", "ocr_failed"}:
            continue
        accepted = bool(group.ocr_text_norm) and (
            group.ocr_confidence >= config.ocr.min_quality_score
            or not config.ocr.reject_non_japanese_noise
        )
        if not accepted:
            group.status = "ocr_rejected"
            group.skip_reason = "empty_ocr"
            if group.ocr_text_norm:
                group.skip_reason = f"low_ocr_quality:{group.ocr_confidence:.3f}"
            continue
        group.status = "ocr_accepted"
        translatable.append(group)
    return translatable


def _apply_translation_batch(
    groups: list[TextGroup],
    translations: ValidatedTranslationBatch,
    config: AppConfig,
) -> None:
    for group in groups:
        raw_translation = translations.by_region_key[group.mapping_region_key]
        translation = sanitize_translation_text(raw_translation, source=group.ocr_text)
        validation = validate_translation(group.ocr_text, translation, config.openrouter)
        group.mapping_chain = translations.chain_for(group.mapping_region_key)
        group.translation = translation if validation.valid else ""
        group.translation_valid = validation.valid
        if validation.valid:
            group.mapping_chain["validated_translation"] = hashlib.sha256(
                group.translation.encode("utf-8")
            ).hexdigest()
            group.status = "ready"
            group.skip_reason = ""
        else:
            group.mapping_chain["validated_translation"] = None
            group.status = "translation_rejected"
            group.skip_reason = ",".join(validation.issues) or "empty_translation"
        console.print(
            f"  [{group.id}] {group.ocr_text} → {group.translation or '[保留原文]'}",
            markup=False,
        )


def _translate_groups(
    groups: list[TextGroup],
    page_id: str,
    config: AppConfig,
    glossary: dict[str, str],
) -> ResultIssue | None:
    translatable = _prepare_translation_groups(groups, page_id, config)

    if not translatable:
        return None

    try:
        _ordered, _texts, request = _build_translation_request(translatable, page_id)
        for group in translatable:
            request_item = request.by_region_key[group.mapping_region_key]
            group.mapping_chain["request_item"] = request_item.item_id
        translations = _request_translations(translatable, page_id, config, glossary)
    except Exception as error:  # noqa: BLE001 - page boundary must preserve source on failure
        console.print(f"[red]本頁翻譯失敗，保留原文：{error}[/]")
        for group in translatable:
            group.translation = ""
            group.translation_valid = False
            group.status = "translation_failed"
            group.skip_reason = str(error)
        code = (
            "translation_mapping_failed"
            if isinstance(error, MappingContractError)
            else "translation_api_failed"
        )
        details: dict[str, object] = {}
        raw_response_refs = getattr(error, "raw_response_refs", ())
        if raw_response_refs:
            details["raw_response_artifacts"] = [
                reference.to_dict() for reference in raw_response_refs
            ]
        return ResultIssue(
            code=code,
            message=str(error),
            stage="translation",
            page_id=page_id,
            details=details,
        )

    _apply_translation_batch(translatable, translations, config)
    return None


def _mapping_snapshots(
    request_groups: list[TextGroup],
    final_groups: list[TextGroup],
) -> list[GroupMappingSnapshot]:
    """Preserve every request outcome while enriching surviving groups downstream."""

    def identity(group: TextGroup) -> tuple[str, str]:
        request_item = group.mapping_chain.get("request_item")
        if isinstance(request_item, str) and request_item:
            return ("request_item", request_item)
        if group.mapping_region_key:
            return ("region", group.mapping_region_key)
        return ("group", group.id)

    tracked = {identity(group): group for group in request_groups}
    for group in final_groups:
        tracked[identity(group)] = group
    return [GroupMappingSnapshot.from_group(group) for group in tracked.values()]


def _group_failure_issues(
    groups: list[TextGroup],
    page_id: str,
) -> list[ResultIssue]:
    failure_kinds = {
        "ocr_failed": ("ocr_group_failed", "ocr"),
        "translation_rejected": ("translation_rejected", "translation"),
        "render_collision_rejected": ("render_collision_rejected", "layout"),
        "layout_rejected": ("layout_rejected", "layout"),
        "layout_collision_rejected": ("layout_collision_rejected", "layout"),
    }
    issues: list[ResultIssue] = []
    for group in groups:
        failure = failure_kinds.get(group.status)
        if failure is None:
            continue
        code, stage = failure
        reason = group.skip_reason or group.status
        issues.append(
            ResultIssue(
                code=code,
                message=reason,
                stage=stage,
                page_id=page_id,
                details={
                    "group_id": group.id,
                    "group_status": group.status,
                    "reason": reason,
                    "region_ids": list(group.region_ids),
                },
            )
        )
    return issues


def _read_stage_state(
    store: JobStore,
    inputs: StageInputs,
    stage: StageName,
) -> PipelineStageState:
    return decode_pipeline_state(
        inputs.upstream[stage],
        expected_stage=stage,
        read_bytes=store.artifacts.read_bytes,
    )


def _source_artifact(inputs: StageInputs) -> ArtifactRef:
    artifacts = inputs.upstream[StageName.SOURCE]
    if len(artifacts) != 1:
        raise ValueError("source stage must provide exactly one artifact")
    return artifacts[0]


def _source_ref_from_extras(extras: dict[str, object]) -> ArtifactRef:
    payload = extras.get("source_artifact")
    if not isinstance(payload, dict):
        raise TypeError("pipeline state is missing its source artifact reference")
    return ArtifactRef.model_validate(payload)


def _decode_source_image(store: JobStore, reference: ArtifactRef) -> np.ndarray:
    raw = store.artifacts.read_bytes(reference.sha256)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("source artifact is not a decodable image")
    return image


def _png_payload(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise ImageEncodeError("stage image could not be encoded as PNG")
    return buffer.tobytes()


def _source_media_type(source_bytes: bytes, image_path: Path) -> str:
    if source_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if source_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if source_bytes.startswith(b"BM"):
        return "image/bmp"
    if source_bytes.startswith(b"RIFF") and source_bytes[8:12] == b"WEBP":
        return "image/webp"
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    return f"image/{suffix}" if suffix else "application/octet-stream"


def _mapping_snapshot_payload(groups: list[TextGroup]) -> list[dict[str, object]]:
    return [GroupMappingSnapshot.from_group(group).to_manifest() for group in groups]


def _mapping_snapshot_identity(payload: dict[str, object]) -> tuple[str, str]:
    chain = payload.get("chain")
    if isinstance(chain, dict):
        request_item = chain.get("request_item")
        if isinstance(request_item, str) and request_item:
            return ("request_item", request_item)
        region_key = chain.get("region")
        if isinstance(region_key, str) and region_key:
            return ("region", region_key)
    return ("group", str(payload.get("group_id", "")))


def _merge_mapping_snapshot_payloads(
    previous: object,
    groups: list[TextGroup],
) -> list[dict[str, object]]:
    tracked: dict[tuple[str, str], dict[str, object]] = {}
    if isinstance(previous, list):
        for item in previous:
            if isinstance(item, dict):
                tracked[_mapping_snapshot_identity(item)] = dict(item)
    for item in _mapping_snapshot_payload(groups):
        tracked[_mapping_snapshot_identity(item)] = item
    return list(tracked.values())


def _layout_plan_payload(plan: TextLayoutPlan) -> dict[str, object]:
    return {
        "bbox": list(plan.bbox),
        "block_height": plan.block_height,
        "block_width": plan.block_width,
        "center_x": plan.center_x,
        "center_y": plan.center_y,
        "chunks": list(plan.chunks),
        "direction": plan.direction,
        "fits": plan.fits,
        "font_size": plan.font_size,
        "primary_step": plan.primary_step,
        "reason": plan.reason,
        "score": plan.score,
        "secondary_step": plan.secondary_step,
    }


def _layout_plan_from_payload(payload: object) -> TextLayoutPlan:
    if not isinstance(payload, dict):
        raise TypeError("layout plan payload must be an object")
    bbox = payload.get("bbox")
    chunks = payload.get("chunks")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(chunks, list):
        raise ValueError("layout plan payload has invalid bbox/chunks")
    return TextLayoutPlan(
        bbox=tuple(int(value) for value in bbox),
        direction=str(payload["direction"]),
        font_size=int(payload["font_size"]),
        chunks=tuple(str(value) for value in chunks),
        primary_step=float(payload["primary_step"]),
        secondary_step=float(payload["secondary_step"]),
        center_x=float(payload["center_x"]),
        center_y=float(payload["center_y"]),
        block_width=float(payload["block_width"]),
        block_height=float(payload["block_height"]),
        fits=bool(payload["fits"]),
        reason=str(payload["reason"]),
        score=float(payload["score"]),
    )


def _persist_provider_raw_artifacts(
    raw_artifacts: dict[str, tuple[RawResponseRef, bytes]],
    *,
    store: JobStore,
    context: StageContext,
) -> list[dict[str, object]]:
    persisted: list[dict[str, object]] = []
    for reference, raw in raw_artifacts.values():
        artifact = store.store_artifact(
            raw,
            media_type=reference.media_type,
            owner_type="provider_raw_response",
            owner_id=(
                f"{context.job_id}:{context.page_id}:{context.fingerprint}:"
                f"{reference.sha256}"
            ),
        )
        if (
            artifact.sha256 != reference.sha256
            or artifact.size_bytes != reference.size_bytes
        ):
            raise ValueError("persisted provider response does not match its bundle")
        persisted.append(artifact.model_dump(mode="json"))
    return persisted


def _build_pipeline_stage_runners(
    *,
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    source_bytes: bytes,
    store: JobStore,
) -> dict[StageName, StageFunction]:
    page_id = hashlib.sha256(source_bytes).hexdigest()
    source_media_type = _source_media_type(source_bytes, image_path)

    def source_stage(_context: StageContext, _inputs: StageInputs) -> StageOutputs:
        return StageOutputs(
            (ArtifactPayload(source_bytes, source_media_type, "source"),)
        )

    def detect_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        reference = _source_artifact(inputs)
        if reference.sha256 != page_id:
            raise ValueError("source stage artifact does not match the page identity")
        image = _decode_source_image(store, reference)
        with profile_span("detection"):
            detection = detect_text_regions(image, config.detection, config.postprocess)
        set_page_profile_metrics(
            page_id,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            detected_groups=len(detection.groups),
        )
        return encode_pipeline_state(
            detection,
            producer_stage=StageName.DETECT,
            extras={"source_artifact": reference.model_dump(mode="json")},
        )

    def adapter_state(
        inputs: StageInputs,
        *,
        stage: StageName,
        adapter: str,
    ) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.DETECT)
        reference = _source_artifact(inputs)
        if _source_ref_from_extras(state.extras) != reference:
            raise ValueError(f"{adapter} stage received inconsistent source lineage")
        return encode_pipeline_state(
            state.detection,
            producer_stage=stage,
            extras={**state.extras, f"{adapter}_adapter": "v0.3.2-pass-through"},
        )

    def style_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        encoded = adapter_state(inputs, stage=StageName.STYLE, adapter="style")
        reference = _source_artifact(inputs)
        return StageOutputs(
            (
                *encoded.artifacts,
                ArtifactPayload(
                    store.artifacts.read_bytes(reference.sha256),
                    reference.media_type,
                    "legacy_layout_reference",
                ),
            )
        )

    def safe_region_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        return adapter_state(
            inputs,
            stage=StageName.SAFE_REGION,
            adapter="safe_region",
        )

    def ocr_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.DETECT)
        reference = _source_artifact(inputs)
        if _source_ref_from_extras(state.extras) != reference:
            raise ValueError("OCR stage received inconsistent source lineage")
        original = _decode_source_image(store, reference)
        detection = state.detection
        regions_by_id = {region.id: region for region in detection.regions_post}
        groups = list(detection.groups)
        if groups:
            initialize_ocr_model()
        for group in groups:
            try:
                with profile_span("ocr_group", group_id=group.id):
                    ocr_result = ocr_group_detailed(
                        image=original,
                        group=group,
                        regions_by_id=regions_by_id,
                        cfg=config.ocr,
                        image_key=f"artifact:{page_id}",
                    )
                group.ocr_text = ocr_result.text
                group.ocr_confidence = ocr_result.confidence
                group.ocr_source = ocr_result.source
                group.ocr_candidates = [
                    candidate.to_dict() for candidate in ocr_result.candidates
                ]
                group_regions = [
                    regions_by_id[region_id]
                    for region_id in group.region_ids
                    if region_id in regions_by_id
                ]
                fallback_only = bool(group_regions) and all(
                    region.source == "mask_fallback" for region in group_regions
                )
                has_pixel_mask = (
                    group.mask is not None
                    and group.mask.size > 0
                    and bool(np.any(group.mask))
                )
                accepted, reason = assess_ocr_result(
                    ocr_result,
                    config.ocr,
                    fallback_only=fallback_only,
                )
                if not has_pixel_mask:
                    accepted, reason = False, "missing_text_mask"
                group.ocr_text_norm = (
                    normalize_ocr_text(ocr_result.text, weak=True) if accepted else ""
                )
                group.status = "ocr_done" if accepted else "ocr_rejected"
                group.skip_reason = "" if accepted else reason
            except OCRInitializationError:
                raise
            except Exception as error:  # noqa: BLE001 - one group must not poison its peers
                group.ocr_text = ""
                group.ocr_text_norm = ""
                group.ocr_confidence = 0.0
                group.ocr_source = "error"
                group.status = "ocr_failed"
                group.skip_reason = str(error)
        detection.groups = groups
        return encode_pipeline_state(
            detection,
            producer_stage=StageName.OCR,
            extras={**state.extras, "ocr_adapter": "v0.3.2-ensemble"},
        )

    def order_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.DETECT)
        reference = _source_artifact(inputs)
        if _source_ref_from_extras(state.extras) != reference:
            raise ValueError("order stage received inconsistent source lineage")
        regions_by_id = {region.id: region for region in state.detection.regions_post}
        state.detection.groups = _refresh_group_order(
            list(state.detection.groups), regions_by_id, config.postprocess
        )
        return encode_pipeline_state(
            state.detection,
            producer_stage=StageName.ORDER,
            extras={
                **state.extras,
                "order_adapter": "v0.3.2-reading-order",
                "reading_order": config.postprocess.reading_order,
                "ordered_region_ids": [
                    list(group.region_ids) for group in state.detection.groups
                ],
            },
        )

    def translate_stage(context: StageContext, inputs: StageInputs) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.OCR)
        order = _read_stage_state(store, inputs, StageName.ORDER)
        if order.extras.get("reading_order") != config.postprocess.reading_order:
            raise ValueError("translation stage received a mismatched order artifact")
        if _source_ref_from_extras(state.extras) != _source_ref_from_extras(order.extras):
            raise ValueError("translation stage received inconsistent source lineage")
        detection = state.detection
        regions_by_id = {region.id: region for region in detection.regions_post}
        groups = _merge_duplicate_groups(list(detection.groups), config.postprocess)
        groups = _refresh_group_order(groups, regions_by_id, config.postprocess)
        translatable = _prepare_translation_groups(groups, page_id, config)
        if translatable:
            _ordered, _texts, request = _build_translation_request(
                translatable, page_id
            )
            for group in translatable:
                group.mapping_chain["request_item"] = request.by_region_key[
                    group.mapping_region_key
                ].item_id

            def fetch() -> bytes:
                try:
                    batch = _request_translations(
                        translatable, page_id, config, glossary
                    )
                except Exception as error:
                    if getattr(error, "raw_response_refs", ()):
                        return _serialize_translation_failure_bundle(
                            request,
                            error,
                            artifact_root=config.paths.output_dir,
                        )
                    raise
                return _serialize_translation_bundle(
                    batch,
                    artifact_root=config.paths.output_dir,
                )

            bundle = context.get_or_fetch_raw_response(request.request_id, fetch)
            try:
                translations, raw_artifacts = _deserialize_translation_bundle(bundle)
            except TranslationBundleReplayError as error:
                raw_artifacts = error.raw_artifacts
                _persist_provider_raw_artifacts(
                    raw_artifacts,
                    store=store,
                    context=context,
                )
                raw_references = tuple(
                    reference for reference, _raw in raw_artifacts.values()
                )
                if error.mapping_issues:
                    raise MappingContractError(
                        error.mapping_issues,
                        raw_response_refs=raw_references,
                    ) from error
                raise RuntimeError(f"{error.error_type}: {error}") from error
            if translations.request != request:
                raise ValueError("replayed translation request does not match this stage")
            persisted = _persist_provider_raw_artifacts(
                raw_artifacts,
                store=store,
                context=context,
            )
            _apply_translation_batch(translatable, translations, config)
        else:
            persisted = []
        detection.groups = groups
        return encode_pipeline_state(
            detection,
            producer_stage=StageName.TRANSLATE,
            extras={
                **state.extras,
                "mapping_snapshots": _mapping_snapshot_payload(groups),
                "provider_response_artifacts": persisted,
            },
        )

    def layout_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.TRANSLATE)
        style = _read_stage_state(store, inputs, StageName.STYLE)
        safe_region = _read_stage_state(store, inputs, StageName.SAFE_REGION)
        reference = _source_ref_from_extras(style.extras)
        if (
            _source_ref_from_extras(state.extras) != reference
            or _source_ref_from_extras(safe_region.extras) != reference
        ):
            raise ValueError("layout stage received inconsistent source lineage")
        if reference not in inputs.upstream[StageName.STYLE]:
            raise ValueError("layout reference is not a declared style-stage artifact")
        original = _decode_source_image(store, reference)
        detection = state.detection
        regions_by_id = {region.id: region for region in detection.regions_post}
        groups = list(detection.groups)
        groups = _merge_translation_duplicates(groups, config.postprocess)
        groups = _resolve_render_collisions(groups, regions_by_id, config.postprocess)
        groups = _refresh_group_order(groups, regions_by_id, config.postprocess)
        with profile_span("layout", group_count=len(groups)):
            plans = _preflight_layout_plans(original, groups, regions_by_id, config)
        _record_mapping_layout_plans(groups)
        detection.groups = groups
        return encode_pipeline_state(
            detection,
            producer_stage=StageName.LAYOUT,
            extras={
                **state.extras,
                "layout_plans": {
                    group_id: _layout_plan_payload(plan)
                    for group_id, plan in sorted(plans.items())
                },
                "mapping_snapshots": _merge_mapping_snapshot_payloads(
                    state.extras.get("mapping_snapshots"), groups
                ),
            },
        )

    def inpaint_render_stage(
        _context: StageContext, inputs: StageInputs
    ) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.LAYOUT)
        reference = _source_artifact(inputs)
        if _source_ref_from_extras(state.extras) != reference:
            raise ValueError("render stage received inconsistent source lineage")
        original = _decode_source_image(store, reference)
        detection = state.detection
        groups = list(detection.groups)
        regions_by_id = {region.id: region for region in detection.regions_post}
        layout_payloads = state.extras.get("layout_plans")
        if not isinstance(layout_payloads, dict):
            raise TypeError("render stage is missing layout plans")
        plans = {
            str(group_id): _layout_plan_from_payload(payload)
            for group_id, payload in layout_payloads.items()
        }
        detection.groups = groups
        with profile_span("inpaint"):
            inpainted = inpaint_regions(original, detection, config.inpainting)
        rendered = inpainted.copy()
        for group in groups:
            if not (
                group.translation_valid
                and group.translation.strip()
                and group.id in plans
            ):
                continue
            with profile_span("render", group_id=group.id):
                rendered = render_text_into_group(
                    image=rendered,
                    group=group,
                    regions_by_id=regions_by_id,
                    text=group.translation,
                    font_path=config.paths.font,
                    cfg=config.typesetting,
                    fallback_font_path=config.paths.font_fallback,
                    layout_plan=plans[group.id],
                    layout_reference_image=original,
                )
            group.mapping_chain["render_target"] = f"render:{group.id}"
        inpainted_raw = _png_payload(inpainted)
        rendered_raw = _png_payload(rendered)
        rendered_sha256 = hashlib.sha256(rendered_raw).hexdigest()
        detection.groups = groups
        encoded_state = encode_pipeline_state(
            detection,
            producer_stage=StageName.INPAINT_RENDER,
            extras={
                **state.extras,
                "inpainted_image_sha256": hashlib.sha256(inpainted_raw).hexdigest(),
                "mapping_snapshots": _merge_mapping_snapshot_payloads(
                    state.extras.get("mapping_snapshots"), groups
                ),
                "rendered_image_sha256": rendered_sha256,
            },
        )
        return StageOutputs(
            (
                *encoded_state.artifacts,
                ArtifactPayload(inpainted_raw, "image/png", "inpainted_page"),
                ArtifactPayload(rendered_raw, "image/png", "rendered_page"),
            )
        )

    def encode_stage(_context: StageContext, inputs: StageInputs) -> StageOutputs:
        state = _read_stage_state(store, inputs, StageName.INPAINT_RENDER)
        rendered_sha256 = state.extras.get("rendered_image_sha256")
        if not isinstance(rendered_sha256, str):
            raise TypeError("render stage state is missing the rendered image hash")
        artifacts = {
            artifact.sha256: artifact
            for artifact in inputs.upstream[StageName.INPAINT_RENDER]
        }
        rendered = artifacts.get(rendered_sha256)
        if rendered is None or rendered.media_type != "image/png":
            raise ValueError("rendered image is not a declared render-stage artifact")
        return StageOutputs(
            (
                ArtifactPayload(
                    store.artifacts.read_bytes(rendered.sha256),
                    "image/png",
                    "encoded_page",
                ),
            )
        )

    return {
        StageName.SOURCE: source_stage,
        StageName.DETECT: detect_stage,
        StageName.STYLE: style_stage,
        StageName.SAFE_REGION: safe_region_stage,
        StageName.OCR: ocr_stage,
        StageName.ORDER: order_stage,
        StageName.TRANSLATE: translate_stage,
        StageName.LAYOUT: layout_stage,
        StageName.INPAINT_RENDER: inpaint_render_stage,
        StageName.ENCODE: encode_stage,
    }


def _mapping_snapshots_from_extras(extras: dict[str, object]) -> list[GroupMappingSnapshot]:
    payloads = extras.get("mapping_snapshots")
    if not isinstance(payloads, list):
        return []
    snapshots: list[GroupMappingSnapshot] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError("mapping snapshot payload must be an object")
        region_ids = payload.get("region_ids")
        chain = payload.get("chain")
        if not isinstance(region_ids, list) or not isinstance(chain, dict):
            raise TypeError("mapping snapshot region_ids/chain have invalid types")
        duplicate_of = payload.get("duplicate_of")
        snapshots.append(
            GroupMappingSnapshot(
                group_id=str(payload["group_id"]),
                region_ids=tuple(str(value) for value in region_ids),
                group_status=str(payload["group_status"]),
                translation_valid=bool(payload["translation_valid"]),
                skip_reason=str(payload["skip_reason"]),
                duplicate_of=(str(duplicate_of) if duplicate_of is not None else None),
                chain=dict(chain),
            )
        )
    return snapshots


def _page_result_from_stage_outcomes(
    *,
    image_path: Path,
    store: JobStore,
    outcomes: dict[StageName, StageOutcome],
) -> PageResult:
    render_state = decode_pipeline_state(
        outcomes[StageName.INPAINT_RENDER].outputs,
        expected_stage=StageName.INPAINT_RENDER,
        read_bytes=store.artifacts.read_bytes,
    )
    encoded_outputs = outcomes[StageName.ENCODE].outputs
    if len(encoded_outputs) != 1 or encoded_outputs[0].media_type != "image/png":
        raise ValueError("encode stage did not produce one PNG artifact")
    encoded_raw = store.artifacts.read_bytes(encoded_outputs[0].sha256)
    image = cv2.imdecode(np.frombuffer(encoded_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("encode stage output is not a decodable PNG")
    original = _decode_source_image(
        store,
        _source_ref_from_extras(render_state.extras),
    )
    detection = render_state.detection
    groups = list(detection.groups)
    detection_issues = [
        ResultIssue(
            code=issue.code,
            message=issue.message,
            stage="detection",
            page_id=outcomes[StageName.SOURCE].outputs[0].sha256,
            details=issue.details,
        )
        for issue in detection.issues
    ]
    group_issues = _group_failure_issues(
        groups,
        outcomes[StageName.SOURCE].outputs[0].sha256,
    )
    blocking = group_issues[0] if group_issues else None
    page_id = outcomes[StageName.SOURCE].outputs[0].sha256
    set_page_profile_metrics(
        page_id,
        final_groups=len(groups),
        renderable_groups=sum(
            group.translation_valid and bool(group.translation.strip()) for group in groups
        ),
    )
    return PageResult(
        page_id=page_id,
        source_path=image_path,
        status="blocked" if blocking is not None else "succeeded",
        image=image,
        source_image=original,
        regions=detection.regions_post,
        ocr_results=[group.ocr_text for group in groups],
        translations=[group.translation for group in groups],
        groups=groups,
        mapping_chains=(
            _mapping_snapshots_from_extras(render_state.extras)
            or [GroupMappingSnapshot.from_group(group) for group in groups]
        ),
        issues=[*detection_issues, *group_issues],
        stage_failure=blocking.stage if blocking is not None else None,
    )


def _dump_debug_artifacts(
    image_path: Path,
    config: AppConfig,
    original_img: np.ndarray,
    detection: DetectionResult,
    groups: list[TextGroup],
    inpainted_img: np.ndarray | None,
    final_img: np.ndarray | None,
) -> None:
    dump_debug_artifacts(
        output_dir=config.paths.output_dir,
        page_name=image_path.name,
        original_image=original_img,
        regions_raw=detection.regions_raw,
        regions_post=detection.regions_post,
        groups=groups,
        save_overlays=True,
        inpainted_image=inpainted_img,
        final_image=final_img,
    )


def _page_id_for_path(image_path: Path) -> str:
    try:
        with image_path.open("rb") as source_file:
            return hashlib.file_digest(source_file, "sha256").hexdigest()
    except OSError:
        return hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()


def _initial_page_document(
    *, image_path: Path, source_bytes: bytes, job_id: str, store: JobStore
) -> PageDocument:
    page_id = hashlib.sha256(source_bytes).hexdigest()
    original_artifact = store.store_artifact(
        source_bytes,
        media_type=_source_media_type(source_bytes, image_path),
        owner_type="source_page",
        owner_id=f"{job_id}:{page_id}:original",
    )
    decoded = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        width = height = 1
        mode = "unknown"
    else:
        height, width = decoded.shape[:2]
        channels = 1 if decoded.ndim == 2 else decoded.shape[2]
        mode = {1: "L", 3: "BGR", 4: "BGRA"}.get(channels, f"channels:{channels}")
    return PageDocument(
        source=SourcePage(
            page_id=page_id,
            original_bytes_sha256=page_id,
            source_path=str(image_path),
            width=int(width),
            height=int(height),
            mode=mode,
            original_artifact=original_artifact,
        )
    )


def _typed_issue(issue: ResultIssue) -> Issue:
    stage = {
        "decode": StageName.SOURCE,
        "detection": StageName.DETECT,
        "ocr": StageName.OCR,
        "translation": StageName.TRANSLATE,
        "layout": StageName.LAYOUT,
        "encode": StageName.ENCODE,
        "output": StageName.ENCODE,
    }.get(issue.stage, StageName.SOURCE)
    code = {
        StageName.DETECT: IssueCode.DETECTOR_FAILED,
        StageName.OCR: IssueCode.OCR_FAILED,
        StageName.TRANSLATE: IssueCode.TRANSLATION_FAILED,
        StageName.LAYOUT: IssueCode.LAYOUT_FAILED,
        StageName.ENCODE: IssueCode.ENCODE_FAILED,
    }.get(stage, IssueCode.SOURCE_FAILED if issue.stage == "decode" else IssueCode.PIPELINE_FAILED)
    if issue.stage == "output":
        code = IssueCode.OUTPUT_FAILED
    return Issue(
        code=code,
        severity=IssueSeverity.ERROR,
        stage=stage,
        message=issue.message,
        page_id=issue.page_id if issue.page_id and len(issue.page_id) == 64 else None,
        details={"legacy_code": issue.code, **issue.details},
    )


def _raw_response_artifact(group: TextGroup) -> ArtifactRef:
    response = group.mapping_chain.get("raw_response_item")
    if not isinstance(response, dict):
        raise TypeError(f"translated group {group.id} has no raw response mapping")
    payload = response.get("artifact")
    if not isinstance(payload, dict):
        raise TypeError(f"translated group {group.id} has no raw response artifact")
    return ArtifactRef(
        sha256=str(payload["sha256"]),
        media_type=str(payload["media_type"]),
        size_bytes=int(payload["size_bytes"]),
    )


def _document_from_page_result(
    *,
    previous: PageDocument,
    page: PageResult,
    outcomes: dict[StageName, StageOutcome],
    store: JobStore,
    job_id: str,
) -> PageDocument:
    observations: list[RegionObservation] = []
    for region in page.regions:
        mask_refs: tuple[ArtifactRef, ...] = ()
        if region.local_mask is not None:
            encoded, buffer = cv2.imencode(".png", region.local_mask)
            if encoded:
                mask = store.store_artifact(
                    buffer.tobytes(),
                    media_type="image/png",
                    owner_type="region_mask",
                    owner_id=f"{job_id}:{page.page_id}:{region.id}",
                )
                mask_refs = (mask,)
        observations.append(
            RegionObservation(
                bbox=BoundingBox(
                    x=float(region.x),
                    y=float(region.y),
                    width=float(region.w),
                    height=float(region.h),
                ),
                detector_score=max(0.0, min(1.0, float(region.confidence))),
                source=region.source,
                raw_index=region.raw_index,
                orientation="vertical" if region.vertical else "horizontal",
                mask_refs=mask_refs,
            )
        )
    reconciled = reconcile_regions(
        page_id=page.page_id,
        detector_fingerprint=outcomes[StageName.DETECT].fingerprint,
        observations=observations,
        previous=previous,
    )
    revision_by_legacy_id = {
        region.id: revision
        for region, revision in zip(page.regions, reconciled.current_revisions, strict=True)
    }
    current_revision_ids = {
        revision.revision_id for revision in reconciled.current_revisions
    }
    ocr_records: list[OCRRecord] = [
        record
        for record in previous.ocr_records
        if record.revision_id not in current_revision_ids
    ]
    translations: list[TranslationRecord] = [
        record
        for record in previous.translations
        if record.revision_id not in current_revision_ids
    ]
    group_issues: list[Issue] = []
    recorded_ocr: set[str] = set()
    recorded_translation: set[str] = set()
    for group in page.groups:
        for legacy_region_id in group.region_ids:
            revision = revision_by_legacy_id.get(legacy_region_id)
            if revision is None:
                continue
            if group.ocr_text and revision.revision_id not in recorded_ocr:
                candidate = OCRCandidate(
                    raw_text=group.ocr_text,
                    normalized_text=group.ocr_text_norm,
                    confidence=max(0.0, min(1.0, float(group.ocr_confidence))),
                    confidence_kind="ensemble",
                    source_view=group.ocr_source or "legacy",
                )
                ocr_records.append(
                    OCRRecord(
                        region_id=revision.region_id,
                        revision_id=revision.revision_id,
                        candidates=(candidate,),
                        selected_index=0,
                        model_revision="kha-white/manga-ocr-base:revision-unpinned",
                        preprocess_version="v0.3.2-ensemble.1",
                    )
                )
                recorded_ocr.add(revision.revision_id)
            if group.translation_valid and revision.revision_id not in recorded_translation:
                request_item = group.mapping_chain.get("request_item")
                if not isinstance(request_item, str) or not request_item:
                    raise ValueError(
                        f"translated group {group.id} has no durable request item identity"
                    )
                translations.append(
                    TranslationRecord(
                        region_id=revision.region_id,
                        revision_id=revision.revision_id,
                        request_item_id=request_item,
                        raw_response_ref=_raw_response_artifact(group),
                        validated_text=group.translation,
                    )
                )
                recorded_translation.add(revision.revision_id)
            elif not group.translation_valid:
                if group.status.startswith("ocr"):
                    code = IssueCode.OCR_REJECTED
                    issue_stage = StageName.OCR
                elif "layout" in group.status or "collision" in group.status:
                    code = IssueCode.LAYOUT_REJECTED
                    issue_stage = StageName.LAYOUT
                else:
                    code = IssueCode.TRANSLATION_REJECTED
                    issue_stage = StageName.TRANSLATE
                group_issues.append(
                    Issue(
                        code=code,
                        severity=IssueSeverity.WARNING,
                        stage=issue_stage,
                        message=group.skip_reason,
                        page_id=page.page_id,
                        region_id=revision.region_id,
                        details={"group_id": group.id, "group_status": group.status},
                    )
                )
    stage_records = _stage_records(outcomes)
    retained_entities = tuple(
        entity
        for entity in previous.entities
        if entity.entity_id != "_page_result" and entity.kind != "mapping_snapshot"
    )
    mapping_entities = tuple(
        EntityRecord(
            entity_id=f"_mapping:{index:04d}",
            kind="mapping_snapshot",
            canonical_name=snapshot.group_status or "unknown",
            attributes=snapshot.to_manifest(),
        )
        for index, snapshot in enumerate(page.mapping_chains)
    )
    return PageDocument(
        source=previous.source,
        region_identities=reconciled.identities,
        region_revisions=reconciled.revisions,
        ocr_records=tuple(ocr_records),
        translations=tuple(translations),
        stages=stage_records,
        issues=tuple(reconciled.issues)
        + tuple(
            _typed_issue(issue)
            for issue in page.issues
            if "group_id" not in issue.details
        )
        + tuple(group_issues),
        entities=retained_entities
        + (
            EntityRecord(
                entity_id="_page_result",
                kind="pipeline_result",
                canonical_name=page.status,
                attributes={"stage_failure": page.stage_failure},
            ),
        )
        + mapping_entities,
    )


def _stage_records(outcomes: dict[StageName, StageOutcome]) -> tuple[StageRecord, ...]:
    return tuple(
        StageRecord(
            stage=name,
            status=StageStatus.SUCCEEDED,
            fingerprint=outcome.fingerprint,
            output_hashes=tuple(item.sha256 for item in outcome.outputs),
            # Cache hits are run-attempt telemetry kept in SQLite.  Persisting
            # them here would change canonical PageDocument bytes on a replay.
            cache_hit=False,
        )
        for name, outcome in outcomes.items()
    )


def _active_region_revisions(document: PageDocument) -> tuple[RegionRevision, ...]:
    revision_by_id = {revision.revision_id: revision for revision in document.region_revisions}
    return tuple(
        revision_by_id[identity.active_revision_id]
        for identity in document.region_identities
        if identity.is_active
    )


def _page_result_from_document(
    document: PageDocument, encoded: ArtifactRef, store: JobStore
) -> PageResult:
    raw = store.artifacts.read_bytes(encoded.sha256)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    entity = next((item for item in document.entities if item.entity_id == "_page_result"), None)
    status = entity.canonical_name if entity is not None else "succeeded"
    if status not in {"succeeded", "failed", "blocked"}:
        status = "failed"
    active_revision_ids = {
        revision.revision_id for revision in _active_region_revisions(document)
    }
    regions = [
        TextRegion(
            id=str(revision.region_id),
            x=int(revision.bbox.x),
            y=int(revision.bbox.y),
            w=int(revision.bbox.width),
            h=int(revision.bbox.height),
            vertical=revision.orientation == "vertical",
            confidence=revision.detector_score,
            source=revision.source,
            raw_index=revision.raw_index,
        )
        for revision in _active_region_revisions(document)
    ]
    return PageResult(
        page_id=document.source.page_id,
        source_path=Path(document.source.source_path),
        status=status,
        image=image,
        regions=regions,
        ocr_results=[
            record.candidates[record.selected_index].raw_text
            for record in document.ocr_records
            if record.selected_index is not None
            and record.revision_id in active_revision_ids
        ],
        translations=[
            record.validated_text
            for record in document.translations
            if record.revision_id in active_revision_ids
        ],
        mapping_chains=_mapping_snapshots_from_extras(
            {
                "mapping_snapshots": [
                    item.attributes
                    for item in document.entities
                    if item.kind == "mapping_snapshot"
                ]
            }
        ),
        issues=[
            ResultIssue(
                code=issue.code.value,
                message=issue.message,
                stage=issue.stage.value,
                page_id=issue.page_id,
                details=issue.details,
            )
            for issue in document.issues
        ],
        stage_failure=(
            str(entity.attributes.get("stage_failure"))
            if entity is not None and entity.attributes.get("stage_failure") is not None
            else None
        ),
    )


def process_single_page_staged(
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    *,
    store: JobStore,
    job_id: str,
    resume: bool,
    force_stage: StageName | None,
    debug: bool,
    dump_json: bool,
    save_intermediate: bool,
    prep_manual: bool,
) -> PageResult:
    source_bytes = image_path.read_bytes()
    page_id = hashlib.sha256(source_bytes).hexdigest()
    previous = store.load_page_document(job_id=job_id, page_id=page_id)
    if previous is None:
        previous = _initial_page_document(
            image_path=image_path, source_bytes=source_bytes, job_id=job_id, store=store
        )
        store.store_page_document(job_id, previous)
    glossary_revision = glossary_fingerprint(glossary)
    config_payload = stage_config(config, glossary_revision)
    runners = _build_pipeline_stage_runners(
        image_path=image_path,
        config=config,
        glossary=glossary,
        source_bytes=source_bytes,
        store=store,
    )
    specs = build_pipeline_stage_specs(
        config=config,
        glossary_revision=glossary_revision,
        runners=runners,
    )
    outcomes = StageRunner(
        store=store,
        job_id=job_id,
        page_id=page_id,
        specs=specs,
        config=config_payload,
    ).run(resume=resume, force_stage=force_stage)
    page = _page_result_from_stage_outcomes(
        image_path=image_path,
        store=store,
        outcomes=outcomes,
    )
    document = _document_from_page_result(
        previous=previous,
        page=page,
        outcomes=outcomes,
        store=store,
        job_id=job_id,
    )
    store.store_page_document(job_id, document)
    if dump_json or prep_manual:
        dump_page_document(config.paths.output_dir, image_path.name, document)
    return page


def process_single_page(
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    debug: bool = False,
    dump_json: bool = False,
    save_intermediate: bool = False,
    prep_manual: bool = False,
    *,
    page_id: str | None = None,
) -> PageResult:
    page_id = page_id or _page_id_for_path(image_path)
    with profile_page(page_id, str(image_path)):
        return _process_single_page_impl(
            image_path,
            config,
            glossary,
            page_id=page_id,
            debug=debug,
            dump_json=dump_json,
            save_intermediate=save_intermediate,
            prep_manual=prep_manual,
        )


def _process_single_page_impl(
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    *,
    page_id: str,
    debug: bool,
    dump_json: bool,
    save_intermediate: bool,
    prep_manual: bool,
) -> PageResult:
    with profile_span("decode"):
        image = read_image(image_path)
    if image is None:
        message = f"無法讀取圖片：{image_path}"
        return PageResult(
            page_id=page_id,
            source_path=image_path,
            status="failed",
            issues=[
                ResultIssue(
                    code="image_read_failed",
                    message=message,
                    stage="decode",
                    page_id=page_id,
                )
            ],
            stage_failure="decode",
        )
    original = image.copy()
    console.print(f"\n[bold]處理：{image_path.name}[/]")

    with profile_span("detection"):
        detection = detect_text_regions(image, config.detection, config.postprocess)
    set_page_profile_metrics(
        page_id,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        detected_groups=len(detection.groups),
    )
    detection_issues = [
        ResultIssue(
            code=issue.code,
            message=issue.message,
            stage="detection",
            page_id=page_id,
            details=issue.details,
        )
        for issue in detection.issues
    ]
    fallback_count = sum(region.source == "mask_fallback" for region in detection.regions_raw)
    console.print(
        f"  raw={len(detection.regions_raw)} post={len(detection.regions_post)} "
        f"groups={len(detection.groups)} mask-fallback={fallback_count}"
    )

    regions_by_id = {region.id: region for region in detection.regions_post}
    groups = list(detection.groups)
    if groups:
        try:
            initialize_ocr_model()
        except OCRInitializationError as error:
            return PageResult(
                page_id=page_id,
                source_path=image_path,
                status="blocked",
                source_image=original,
                regions=detection.regions_post,
                mapping_chains=[GroupMappingSnapshot.from_group(group) for group in groups],
                issues=[
                    *detection_issues,
                    ResultIssue(
                        code="ocr_initialization_failed",
                        message=str(error),
                        stage="ocr",
                        page_id=page_id,
                    )
                ],
                stage_failure="ocr",
            )

    for group in groups:
        try:
            with profile_span("ocr_group", group_id=group.id):
                ocr_result = ocr_group_detailed(
                    image=original,
                    group=group,
                    regions_by_id=regions_by_id,
                    cfg=config.ocr,
                    image_key=str(image_path.resolve()),
                )
            group.ocr_text = ocr_result.text
            group.ocr_confidence = ocr_result.confidence
            group.ocr_source = ocr_result.source
            group.ocr_candidates = [candidate.to_dict() for candidate in ocr_result.candidates]
            group_regions = [
                regions_by_id[rid]
                for rid in group.region_ids
                if rid in regions_by_id
            ]
            fallback_only = bool(group_regions) and all(
                region.source == "mask_fallback" for region in group_regions
            )
            has_pixel_mask = (
                group.mask is not None
                and group.mask.size > 0
                and bool(np.any(group.mask))
            )
            accepted, reason = assess_ocr_result(
                ocr_result,
                config.ocr,
                fallback_only=fallback_only,
            )
            if not has_pixel_mask:
                accepted, reason = False, "missing_text_mask"

            group.ocr_text_norm = (
                normalize_ocr_text(ocr_result.text, weak=True) if accepted else ""
            )
            group.status = "ocr_done" if accepted else "ocr_rejected"
            group.skip_reason = "" if accepted else reason
        except OCRInitializationError:
            raise
        except Exception as error:  # noqa: BLE001 - isolate OCR failure to this region
            group.ocr_text = ""
            group.ocr_text_norm = ""
            group.ocr_confidence = 0.0
            group.ocr_source = "error"
            group.status = "ocr_failed"
            group.skip_reason = str(error)
            console.print(f"[yellow]  [{group.id}] OCR 失敗，保留原文：{error}[/]")

    groups = _merge_duplicate_groups(groups, config.postprocess)
    groups = _refresh_group_order(groups, regions_by_id, config.postprocess)
    with profile_span("translation", group_count=len(groups)):
        translation_issue = _translate_groups(groups, page_id, config, glossary)
    mapping_outcomes = list(groups)
    groups_after_translation = _merge_translation_duplicates(groups, config.postprocess)
    if len(groups_after_translation) != len(groups):
        console.print(
            f"[yellow]  翻譯後再合併 {len(groups) - len(groups_after_translation)} 個強重疊重複框[/]"
        )
    groups_after_collision = _resolve_render_collisions(
        groups_after_translation,
        regions_by_id,
        config.postprocess,
    )
    collision_count = sum(
        group.status == "render_collision_rejected"
        for group in groups_after_collision
    )
    if collision_count:
        console.print(
            f"[yellow]  阻止 {collision_count} 個強重疊譯文寫入同一位置[/]"
        )
    groups = _refresh_group_order(
        groups_after_collision,
        regions_by_id,
        config.postprocess,
    )

    # 排版先在原圖上完整預演。放不下、會縮得過小或實際文字塊會互撞時，
    # 直接保留原文；不能先擦掉再發現無法安全寫回。
    with profile_span("layout", group_count=len(groups)):
        layout_plans = _preflight_layout_plans(
            original,
            groups,
            regions_by_id,
            config,
        )
    _record_mapping_layout_plans(groups)
    layout_rejected = sum(
        group.status in {"layout_rejected", "layout_collision_rejected"}
        for group in groups
    )
    if layout_rejected:
        console.print(
            f"[yellow]  {layout_rejected} 個譯文無法以接近原字級安全排版，已保留原文[/]"
        )

    # Inpainter 會依 translation_valid 過濾；OCR／翻譯／排版失敗的原文都不會被擦掉。
    detection.groups = groups
    with profile_span("inpaint"):
        inpainted = inpaint_regions(original, detection, config.inpainting)
    result = inpainted.copy()

    renderable = [
        group
        for group in groups
        if group.translation_valid and group.translation.strip() and group.id in layout_plans
    ]
    for group in renderable:
        with profile_span("render", group_id=group.id):
            result = render_text_into_group(
                image=result,
                group=group,
                regions_by_id=regions_by_id,
                text=group.translation,
                font_path=config.paths.font,
                cfg=config.typesetting,
                fallback_font_path=config.paths.font_fallback,
                layout_plan=layout_plans[group.id],
                layout_reference_image=original,
            )
        group.mapping_chain["render_target"] = f"render:{group.id}"

    unresolved = [group for group in groups if not group.translation_valid]
    if unresolved:
        console.print(
            f"[yellow]  {len(unresolved)} 個候選未通過 OCR／翻譯／排版檢查，已保留原文；"
            "可用 --debug --dump-json 查看原因。[/]"
        )

    if prep_manual or save_intermediate:
        intermediate_dir = config.paths.output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        write_image(intermediate_dir / f"{image_path.stem}_original.png", original)
        write_image(intermediate_dir / f"{image_path.stem}_inpainted.png", inpainted)
        write_image(intermediate_dir / f"{image_path.stem}_blanked.png", inpainted)

    if debug or dump_json or prep_manual:
        _dump_debug_artifacts(
            image_path=image_path,
            config=config,
            original_img=original,
            detection=detection,
            groups=groups,
            inpainted_img=inpainted if (debug or save_intermediate or prep_manual) else None,
            final_img=result if debug else None,
        )

    group_issues = _group_failure_issues(groups, page_id)
    issues = [*detection_issues, *group_issues]
    if translation_issue is not None:
        issues.append(translation_issue)
    blocking_issue = translation_issue or (group_issues[0] if group_issues else None)
    set_page_profile_metrics(page_id, final_groups=len(groups), renderable_groups=len(renderable))
    return PageResult(
        page_id=page_id,
        source_path=image_path,
        status="blocked" if blocking_issue is not None else "succeeded",
        image=result,
        source_image=original,
        regions=detection.regions_post,
        ocr_results=[group.ocr_text for group in groups],
        translations=[group.translation for group in groups],
        groups=groups,
        mapping_chains=_mapping_snapshots(mapping_outcomes, groups),
        issues=issues,
        stage_failure=blocking_issue.stage if blocking_issue is not None else None,
    )


def run_pipeline(
    config: AppConfig,
    debug: bool = False,
    dump_json: bool = False,
    save_intermediate: bool = False,
    prep_manual: bool = False,
    resume: bool = False,
    force_stage: StageName | None = None,
    job_id: str | None = None,
    state_dir: Path | None = None,
) -> BatchResult:
    output_dir = config.paths.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        issue = ResultIssue(code="output_write_failed", message=str(error), stage="output")
        return BatchResult(status="failed", pages=[], issues=[issue])

    image_files = get_image_files(config.paths.input_dir)
    if not image_files:
        message = f"在 {config.paths.input_dir} 找不到任何圖片檔"
        console.print(f"[red]{message}[/]")
        batch = BatchResult(
            status="failed",
            pages=[],
            issues=[ResultIssue(code="no_input_files", message=message, stage="input")],
        )
        _write_batch_manifest(batch, output_dir)
        return batch

    glossary = load_glossary(config.paths.glossary)
    console.print(f"[bold]找到 {len(image_files)} 張圖片[/]")
    console.print(f"[bold]模型：{config.openrouter.model}[/]")
    console.print(f"[bold]輸出：{output_dir}[/]")

    durable_store: JobStore | None = None
    if job_id is not None:
        durable_root = (state_dir or (output_dir / ".manga-translator")).resolve()
        durable_store = JobStore(
            durable_root / "jobs.sqlite3", ArtifactStore(durable_root / "artifacts")
        )
        durable_store.ensure_job(
            job_id,
            config={"model": config.openrouter.model, "input_dir": str(config.paths.input_dir)},
        )
    try:
        if durable_store is None:
            initialize_ocr_model()
    except OCRInitializationError as error:
        pages = [
            PageResult(
                page_id=_page_id_for_path(image_path),
                source_path=image_path,
                status="blocked",
                issues=[
                    ResultIssue(
                        code="ocr_initialization_failed",
                        message=str(error),
                        stage="ocr",
                        page_id=_page_id_for_path(image_path),
                    )
                ],
                stage_failure="ocr",
            )
            for image_path in image_files
        ]
        for page in pages:
            _preserve_failed_source(page, output_dir)
        batch = BatchResult(status=derive_batch_status(pages), pages=pages)
        _write_batch_manifest(batch, output_dir)
        console.print(f"[red]OCR 初始化失敗：{error}[/]")
        return batch

    pages: list[PageResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("翻譯中...", total=len(image_files))
        for image_path in image_files:
            page_id = _page_id_for_path(image_path)
            with profile_page(page_id, str(image_path)):
                try:
                    if durable_store is None:
                        page = process_single_page(
                            image_path=image_path,
                            config=config,
                            glossary=glossary,
                            debug=debug,
                            dump_json=dump_json,
                            save_intermediate=save_intermediate,
                            prep_manual=prep_manual,
                            page_id=page_id,
                        )
                    else:
                        page = process_single_page_staged(
                            image_path=image_path,
                            config=config,
                            glossary=glossary,
                            store=durable_store,
                            job_id=job_id,
                            resume=resume,
                            force_stage=force_stage,
                            debug=debug,
                            dump_json=dump_json,
                            save_intermediate=save_intermediate,
                            prep_manual=prep_manual,
                        )
                except OCRInitializationError as error:
                    page = _failed_page_result(
                        image_path,
                        code="ocr_initialization_failed",
                        stage="ocr",
                        error=error,
                        blocked=True,
                    )
                except Exception as error:  # noqa: BLE001 - page boundary records typed failure
                    page = _failed_page_result(
                        image_path,
                        code="page_processing_failed",
                        stage="pipeline",
                        error=error,
                    )
                _persist_page_result(page, output_dir)
            page.image = None
            page.source_image = None
            pages.append(page)
            progress.advance(task)

    if durable_store is not None:
        durable_store.close()

    batch = BatchResult(status=derive_batch_status(pages), pages=pages)
    _write_batch_manifest(batch, output_dir)
    total_regions = sum(len(page.regions) for page in pages)
    console.print(
        f"\n[bold]批次狀態：{batch.status}；共處理 {len(pages)} 頁，"
        f"{total_regions} 個文字區域，失敗／阻塞 {len(batch.failed_pages)} 頁[/]"
    )
    console.print(f"[bold]輸出目錄：{output_dir}[/]")
    return batch


def _failed_page_result(
    image_path: Path,
    *,
    code: str,
    stage: str,
    error: Exception,
    blocked: bool = False,
) -> PageResult:
    page_id = _page_id_for_path(image_path)
    return PageResult(
        page_id=page_id,
        source_path=image_path,
        status="blocked" if blocked else "failed",
        issues=[ResultIssue(code=code, message=str(error), stage=stage, page_id=page_id)],
        stage_failure=stage,
    )


def _failed_output_path(output_dir: Path, source_path: Path) -> Path:
    return output_dir / "failed" / f"{source_path.stem}.source-preserved{source_path.suffix}"


def _preserve_failed_source(page: PageResult, output_dir: Path) -> None:
    fallback_path = _failed_output_path(output_dir, page.source_path)
    try:
        (output_dir / page.source_path.name).unlink(missing_ok=True)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(page.source_path, fallback_path)
    except OSError as error:
        page.status = "failed"
        page.stage_failure = "output"
        page.issues.append(
            ResultIssue(
                code="output_write_failed",
                message=str(error),
                stage="output",
                page_id=page.page_id,
            )
        )
        page.output_path = None
        page.source_preserved = False
        return
    page.output_path = fallback_path
    page.source_preserved = True


def _persist_page_result(page: PageResult, output_dir: Path) -> None:
    if not page.succeeded:
        _preserve_failed_source(page, output_dir)
        return

    output_path = output_dir / page.source_path.name
    try:
        if page.image is None:
            raise ImageEncodeError(f"沒有可編碼的結果圖片：{page.source_path}")
        with profile_span("encode", output_name=output_path.name):
            write_image_or_raise(output_path, page.image)
    except (ImageEncodeError, ImageWriteError) as error:
        code = "image_encode_failed" if isinstance(error, ImageEncodeError) else "output_write_failed"
        page.status = "failed"
        page.stage_failure = "encode" if isinstance(error, ImageEncodeError) else "output"
        page.issues.append(
            ResultIssue(
                code=code,
                message=str(error),
                stage=page.stage_failure,
                page_id=page.page_id,
            )
        )
        _preserve_failed_source(page, output_dir)
        return
    page.output_path = output_path


def _write_batch_manifest(batch: BatchResult, output_dir: Path) -> None:
    manifest_path = output_dir / "batch-manifest.json"
    temporary_path = output_dir / ".batch-manifest.json.tmp"
    try:
        temporary_path.write_text(
            json.dumps(batch.to_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        batch.issues.append(
            ResultIssue(code="output_write_failed", message=str(error), stage="manifest")
        )
        batch.status = "failed"
        batch.manifest_path = None
        return
    batch.manifest_path = manifest_path
