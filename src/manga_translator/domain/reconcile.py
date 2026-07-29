"""Conservative reconciliation of detector revisions to persistent region identities."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from .ids import dhash_distance, region_id_for_revision, revision_id_for
from .issues import Issue, IssueCode, IssueSeverity, StageName
from .models import (
    ArtifactRef,
    BoundingBox,
    Lineage,
    PageDocument,
    Polygon,
    RegionIdentity,
    RegionRevision,
)


@dataclass(frozen=True)
class RegionObservation:
    bbox: BoundingBox
    detector_score: float
    source: str
    raw_index: int
    polygon: Polygon | None = None
    line_polygons: tuple[Polygon, ...] = ()
    angle_degrees: float = 0.0
    orientation: str = "unknown"
    mask_refs: tuple[ArtifactRef, ...] = ()
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    crop_dhash: int | None = None


@dataclass(frozen=True)
class ReconciliationConfig:
    match_threshold: float = 0.67
    ambiguity_margin: float = 0.08
    lineage_overlap_threshold: float = 0.25
    mask_weight: float = 0.35
    geometry_weight: float = 0.30
    center_weight: float = 0.20
    crop_weight: float = 0.15
    content_match_threshold: float = 0.80
    content_conflict_margin: float = 0.25


@dataclass(frozen=True)
class ReconciliationResult:
    identities: tuple[RegionIdentity, ...]
    revisions: tuple[RegionRevision, ...]
    issues: tuple[Issue, ...]
    current_region_ids: tuple[UUID, ...] = ()
    current_revision_ids: tuple[str, ...] = ()

    @property
    def current_identities(self) -> tuple[RegionIdentity, ...]:
        by_id = {item.region_id: item for item in self.identities}
        return tuple(by_id[item] for item in self.current_region_ids)

    @property
    def current_revisions(self) -> tuple[RegionRevision, ...]:
        by_id = {item.revision_id: item for item in self.revisions}
        return tuple(by_id[item] for item in self.current_revision_ids)


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0.0, min(left.right, right.right) - max(left.x, right.x))
    intersection_height = max(0.0, min(left.bottom, right.bottom) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _polygon_iou(left: Polygon | None, right: Polygon | None) -> float | None:
    if left is None or right is None:
        return None
    left_shape = ShapelyPolygon([(point.x, point.y) for point in left.points])
    right_shape = ShapelyPolygon([(point.x, point.y) for point in right.points])
    if not left_shape.is_valid or not right_shape.is_valid:
        return None
    union = left_shape.union(right_shape).area
    return left_shape.intersection(right_shape).area / union if union > 0 else 0.0


def _mask_iou(
    left: np.ndarray | None,
    left_bbox: BoundingBox,
    right: np.ndarray | None,
    right_bbox: BoundingBox,
) -> float | None:
    if (
        left is None
        or right is None
        or left.ndim != 2
        or right.ndim != 2
        or left.size == 0
        or right.size == 0
    ):
        return None

    def rasterize(mask: np.ndarray, bbox: BoundingBox) -> tuple[np.ndarray, tuple[int, ...]]:
        x1 = math.floor(bbox.x)
        y1 = math.floor(bbox.y)
        x2 = math.ceil(bbox.right)
        y2 = math.ceil(bbox.bottom)
        target_height = max(1, y2 - y1)
        target_width = max(1, x2 - x1)
        rows = np.minimum(
            np.arange(target_height) * mask.shape[0] // target_height,
            mask.shape[0] - 1,
        )
        columns = np.minimum(
            np.arange(target_width) * mask.shape[1] // target_width,
            mask.shape[1] - 1,
        )
        return mask.astype(bool, copy=False)[np.ix_(rows, columns)], (x1, y1, x2, y2)

    left_bool, (left_x1, left_y1, left_x2, left_y2) = rasterize(left, left_bbox)
    right_bool, (right_x1, right_y1, right_x2, right_y2) = rasterize(right, right_bbox)
    overlap_x1 = max(left_x1, right_x1)
    overlap_y1 = max(left_y1, right_y1)
    overlap_x2 = min(left_x2, right_x2)
    overlap_y2 = min(left_y2, right_y2)
    intersection = 0
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        left_overlap = left_bool[
            overlap_y1 - left_y1 : overlap_y2 - left_y1,
            overlap_x1 - left_x1 : overlap_x2 - left_x1,
        ]
        right_overlap = right_bool[
            overlap_y1 - right_y1 : overlap_y2 - right_y1,
            overlap_x1 - right_x1 : overlap_x2 - right_x1,
        ]
        intersection = int(np.logical_and(left_overlap, right_overlap).sum())
    union = int(left_bool.sum()) + int(right_bool.sum()) - intersection
    return intersection / union if union else 0.0


def _center_similarity(left: BoundingBox, right: BoundingBox) -> float:
    left_center = (left.x + left.width / 2, left.y + left.height / 2)
    right_center = (right.x + right.width / 2, right.y + right.height / 2)
    distance = math.dist(left_center, right_center)
    scale = max(math.hypot(left.width, left.height), math.hypot(right.width, right.height), 1.0)
    return max(0.0, 1.0 - distance / scale)


def _similarity(
    current: RegionObservation,
    previous: RegionRevision,
    *,
    previous_mask: np.ndarray | None,
    previous_crop_dhash: int | None,
    config: ReconciliationConfig,
) -> tuple[float, float]:
    mask_score = _mask_iou(current.mask, current.bbox, previous_mask, previous.bbox)
    polygon_score = _polygon_iou(current.polygon, previous.polygon)
    geometry_score = polygon_score if polygon_score is not None else _bbox_iou(
        current.bbox, previous.bbox
    )
    scores: list[tuple[float, float]] = [
        (config.geometry_weight, geometry_score),
        (config.center_weight, _center_similarity(current.bbox, previous.bbox)),
    ]
    if mask_score is not None:
        scores.append((config.mask_weight, mask_score))
    if current.crop_dhash is not None and previous_crop_dhash is not None:
        scores.append(
            (config.crop_weight, 1.0 - dhash_distance(current.crop_dhash, previous_crop_dhash))
        )
    weight = sum(item[0] for item in scores)
    lineage_overlap = max(geometry_score, mask_score if mask_score is not None else 0.0)
    return sum(item_weight * score for item_weight, score in scores) / weight, lineage_overlap


def _lineage_kind(
    current_index: int,
    overlaps: list[list[tuple[int, float]]],
    *,
    threshold: float,
) -> tuple[str | None, tuple[int, ...]]:
    parents = tuple(index for index, overlap in overlaps[current_index] if overlap >= threshold)
    if len(parents) > 1:
        return "merge", parents
    if len(parents) == 1:
        parent_index = parents[0]
        children = sum(
            1
            for candidates in overlaps
            if any(index == parent_index and overlap >= threshold for index, overlap in candidates)
        )
        if children > 1:
            return "split", parents
    return None, ()


def reconcile_regions(
    *,
    page_id: str,
    detector_fingerprint: str,
    observations: Sequence[RegionObservation],
    previous: PageDocument | None = None,
    previous_masks: Mapping[str, np.ndarray] | None = None,
    previous_crop_dhashes: Mapping[str, int] | None = None,
    config: ReconciliationConfig | None = None,
    id_factory: Callable[[], UUID] | None = None,
) -> ReconciliationResult:
    """Assign durable IDs without guessing when detector matches are ambiguous."""

    if previous is not None and previous.source.page_id != page_id:
        raise ValueError("cannot reconcile regions across different source pages")
    config = config or ReconciliationConfig()
    previous_masks = previous_masks or {}
    previous_crop_dhashes = previous_crop_dhashes or {}
    previous_identities = {item.region_id: item for item in previous.region_identities} if previous else {}
    previous_revisions = (
        [
            revision
            for identity in previous.region_identities
            if identity.is_active
            if (revision := next(
                (
                    candidate
                    for candidate in previous.region_revisions
                    if candidate.revision_id == identity.active_revision_id
                ),
                None,
            ))
            is not None
        ]
        if previous
        else []
    )

    revision_ids = [
        revision_id_for(
            page_id=page_id,
            detector_fingerprint=detector_fingerprint,
            bbox=item.bbox,
            polygon=item.polygon,
            line_polygons=item.line_polygons,
            angle_degrees=item.angle_degrees,
            orientation=item.orientation,
            mask_refs=item.mask_refs,
        )
        for item in observations
    ]
    exact = {item.revision_id: item for item in previous_revisions}
    score_rows: list[list[tuple[int, float]]] = []
    content_rows: list[list[tuple[int, float]]] = []
    overlap_rows: list[list[tuple[int, float]]] = []
    for observation in observations:
        scores: list[tuple[int, float]] = []
        content_scores: list[tuple[int, float]] = []
        overlaps: list[tuple[int, float]] = []
        for index, old_revision in enumerate(previous_revisions):
            score, overlap = _similarity(
                observation,
                old_revision,
                previous_mask=previous_masks.get(old_revision.revision_id),
                previous_crop_dhash=previous_crop_dhashes.get(old_revision.revision_id),
                config=config,
            )
            scores.append((index, score))
            overlaps.append((index, overlap))
            previous_dhash = previous_crop_dhashes.get(old_revision.revision_id)
            if observation.crop_dhash is not None and previous_dhash is not None:
                content_scores.append(
                    (index, 1.0 - dhash_distance(observation.crop_dhash, previous_dhash))
                )
        score_rows.append(sorted(scores, key=lambda item: (-item[1], item[0])))
        content_rows.append(sorted(content_scores, key=lambda item: (-item[1], item[0])))
        overlap_rows.append(overlaps)

    proposed: dict[int, int] = {}
    ambiguous: dict[int, tuple[int, ...]] = {}
    for current_index, (revision_id, candidates, content_candidates) in enumerate(
        zip(revision_ids, score_rows, content_rows)
    ):
        exact_revision = exact.get(revision_id)
        if exact_revision is not None:
            proposed[current_index] = previous_revisions.index(exact_revision)
            continue
        if not candidates or candidates[0][1] < config.match_threshold:
            continue
        runner_up = candidates[1][1] if len(candidates) > 1 else 0.0
        if candidates[0][1] - runner_up < config.ambiguity_margin:
            ambiguous[current_index] = tuple(
                index
                for index, score in candidates
                if candidates[0][1] - score < config.ambiguity_margin
            )
            continue
        if content_candidates and content_candidates[0][0] != candidates[0][0]:
            selected_content_score = dict(content_candidates).get(candidates[0][0], 0.0)
            best_content_score = content_candidates[0][1]
            if (
                best_content_score >= config.content_match_threshold
                or best_content_score - selected_content_score
                >= config.content_conflict_margin
            ):
                ambiguous[current_index] = tuple(
                    dict.fromkeys((candidates[0][0], content_candidates[0][0]))
                )
                continue
        proposed[current_index] = candidates[0][0]

    owners: dict[int, list[int]] = {}
    for current_index, previous_index in proposed.items():
        owners.setdefault(previous_index, []).append(current_index)
    for previous_index, current_indices in owners.items():
        if len(current_indices) > 1:
            for current_index in current_indices:
                proposed.pop(current_index, None)
                ambiguous[current_index] = (previous_index,)

    current_identities: list[RegionIdentity] = []
    current_revisions: list[RegionRevision] = []
    issues: list[Issue] = []
    for index, (observation, revision_id) in enumerate(zip(observations, revision_ids)):
        lineage_kind, lineage_indexes = _lineage_kind(
            index,
            overlap_rows,
            threshold=config.lineage_overlap_threshold,
        )
        previous_index = proposed.get(index) if lineage_kind is None else None
        if previous_index is not None:
            region_id = previous_revisions[previous_index].region_id
            lineage = previous_identities[region_id].lineage
        else:
            region_id = (
                id_factory()
                if id_factory is not None
                else region_id_for_revision(page_id=page_id, revision_id=revision_id)
            )
            predecessor_indexes = ambiguous.get(index, ())
            predecessors = tuple(previous_revisions[item].region_id for item in predecessor_indexes)
            lineage_ids = tuple(previous_revisions[item].region_id for item in lineage_indexes)
            lineage = Lineage(
                parents=lineage_ids,
                supersedes=lineage_ids,
                possible_predecessors=predecessors,
                reason=lineage_kind or ("ambiguous_match" if predecessors else "new_detection"),
            )
            if predecessors:
                issues.append(
                    Issue(
                        code=IssueCode.AMBIGUOUS_IDENTITY,
                        severity=IssueSeverity.WARNING,
                        stage=StageName.DETECT,
                        page_id=page_id,
                        region_id=region_id,
                        message="region identity was not reused because the match was ambiguous",
                        details={"possible_predecessors": [str(item) for item in predecessors]},
                    )
                )
        revision = RegionRevision(
            revision_id=revision_id,
            region_id=region_id,
            bbox=observation.bbox,
            polygon=observation.polygon,
            line_polygons=observation.line_polygons,
            angle_degrees=observation.angle_degrees,
            orientation=observation.orientation,
            detector_score=observation.detector_score,
            mask_refs=observation.mask_refs,
            source=observation.source,
            raw_index=observation.raw_index,
        )
        current_identities.append(
            RegionIdentity(
                region_id=region_id,
                active_revision_id=revision_id,
                lineage=lineage,
                is_active=True,
            )
        )
        current_revisions.append(revision)

    identity_history = [item.model_copy(update={"is_active": False}) for item in previous_identities.values()]
    identity_indexes = {item.region_id: index for index, item in enumerate(identity_history)}
    for identity in current_identities:
        existing_index = identity_indexes.get(identity.region_id)
        if existing_index is None:
            identity_indexes[identity.region_id] = len(identity_history)
            identity_history.append(identity)
        else:
            identity_history[existing_index] = identity

    revision_history = list(previous.region_revisions) if previous is not None else []
    known_revision_ids = {item.revision_id for item in revision_history}
    for revision in current_revisions:
        if revision.revision_id not in known_revision_ids:
            revision_history.append(revision)
            known_revision_ids.add(revision.revision_id)

    return ReconciliationResult(
        identities=tuple(identity_history),
        revisions=tuple(revision_history),
        issues=tuple(issues),
        current_region_ids=tuple(item.region_id for item in current_identities),
        current_revision_ids=tuple(item.revision_id for item in current_revisions),
    )


def trace_ancestors(identities: Sequence[RegionIdentity], region_id: UUID) -> tuple[UUID, ...]:
    """Return all reachable lineage ancestors in deterministic order."""

    by_id = {item.region_id: item for item in identities}
    root = by_id.get(region_id)
    if root is None:
        return ()
    seen: set[UUID] = set()
    pending = list(root.lineage.parents)
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        identity = by_id.get(current)
        if identity is not None:
            pending.extend(identity.lineage.parents)
    return tuple(sorted(seen, key=str))
