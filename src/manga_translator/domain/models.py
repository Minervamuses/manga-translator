"""Canonical PageDocument schema; this module never imports legacy pipeline models."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .issues import Issue, StageName, StageStatus, ensure_json_object

SCHEMA_VERSION = "1.0"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Channel = Annotated[int, Field(ge=0, le=255)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)


class ArtifactRef(DomainModel):
    sha256: Sha256
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class Point(DomainModel):
    x: float
    y: float


class BoundingBox(DomainModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


class Polygon(DomainModel):
    points: tuple[Point, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_area(self) -> Polygon:
        coordinates = tuple((point.x, point.y) for point in self.points)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("polygon vertices must be distinct and must not repeat the first point")
        area = 0.0
        for index, point in enumerate(self.points):
            following = self.points[(index + 1) % len(self.points)]
            area += point.x * following.y - following.x * point.y
        if abs(area) <= 1e-9:
            raise ValueError("polygon must have non-zero area")
        if _polygon_has_self_intersection(coordinates):
            raise ValueError("polygon must not self-intersect")
        return self


def _polygon_has_self_intersection(points: tuple[tuple[float, float], ...]) -> bool:
    def cross(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    def on_segment(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
            and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
        )

    def intersects(
        left_start: tuple[float, float],
        left_end: tuple[float, float],
        right_start: tuple[float, float],
        right_end: tuple[float, float],
    ) -> bool:
        values = (
            cross(left_start, left_end, right_start),
            cross(left_start, left_end, right_end),
            cross(right_start, right_end, left_start),
            cross(right_start, right_end, left_end),
        )
        if (values[0] > 0 > values[1] or values[0] < 0 < values[1]) and (
            values[2] > 0 > values[3] or values[2] < 0 < values[3]
        ):
            return True
        return any(
            value == 0.0 and on_segment(segment_start, segment_end, point)
            for value, segment_start, segment_end, point in (
                (values[0], left_start, left_end, right_start),
                (values[1], left_start, left_end, right_end),
                (values[2], right_start, right_end, left_start),
                (values[3], right_start, right_end, left_end),
            )
        )

    edge_count = len(points)
    for left_index in range(edge_count):
        left_end = (left_index + 1) % edge_count
        for right_index in range(left_index + 1, edge_count):
            right_end = (right_index + 1) % edge_count
            if left_index in (right_index, right_end) or left_end in (right_index, right_end):
                continue
            if intersects(
                points[left_index],
                points[left_end],
                points[right_index],
                points[right_end],
            ):
                return True
    return False


class SourcePage(DomainModel):
    page_id: Sha256
    original_bytes_sha256: Sha256
    source_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str = Field(min_length=1)
    exif_orientation: int | None = Field(default=None, ge=1, le=8)
    icc_profile_sha256: Sha256 | None = None
    original_artifact: ArtifactRef

    @model_validator(mode="after")
    def page_id_is_source_hash(self) -> SourcePage:
        if self.page_id != self.original_bytes_sha256:
            raise ValueError("page_id must equal original_bytes_sha256")
        return self


class Lineage(DomainModel):
    parents: tuple[UUID, ...] = ()
    supersedes: tuple[UUID, ...] = ()
    possible_predecessors: tuple[UUID, ...] = ()
    reason: str | None = None


class RegionIdentity(DomainModel):
    region_id: UUID
    active_revision_id: Sha256
    lineage: Lineage = Field(default_factory=Lineage)


class RegionRevision(DomainModel):
    revision_id: Sha256
    region_id: UUID
    bbox: BoundingBox
    polygon: Polygon | None = None
    line_polygons: tuple[Polygon, ...] = ()
    angle_degrees: float = 0.0
    orientation: Literal["horizontal", "vertical", "rotated", "unknown"] = "unknown"
    detector_score: Score
    mask_refs: tuple[ArtifactRef, ...] = ()
    source: str = Field(min_length=1)
    raw_index: int = Field(ge=-1)


class OCRCandidate(DomainModel):
    raw_text: str
    normalized_text: str
    token_scores: tuple[Score, ...] = ()
    confidence: Score
    confidence_kind: Literal["model", "heuristic", "ensemble", "unknown"]
    source_view: str = Field(min_length=1)


class OCRRecord(DomainModel):
    region_id: UUID
    revision_id: Sha256
    candidates: tuple[OCRCandidate, ...]
    selected_index: int | None = Field(default=None, ge=0)
    model_revision: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def selected_candidate_exists(self) -> OCRRecord:
        if self.selected_index is not None and self.selected_index >= len(self.candidates):
            raise ValueError("selected_index outside candidates")
        return self


class StyleFingerprint(DomainModel):
    region_id: UUID
    revision_id: Sha256
    fill_rgb: tuple[Channel, Channel, Channel] | None = None
    stroke_rgb: tuple[Channel, Channel, Channel] | None = None
    shadow_rgb: tuple[Channel, Channel, Channel] | None = None
    stroke_width: float | None = Field(default=None, ge=0)
    ink_density: Score | None = None
    angle_degrees: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)
    confidence: Score


class TranslationRecord(DomainModel):
    region_id: UUID
    revision_id: Sha256
    request_item_id: str = Field(min_length=1)
    raw_response_ref: ArtifactRef
    validated_text: str
    entities: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()


class ShapedRun(DomainModel):
    text: str
    font_sha256: Sha256
    glyph_ids: tuple[int, ...]
    advances: tuple[float, ...]
    offsets: tuple[tuple[float, float], ...] = ()

    @model_validator(mode="after")
    def aligned_glyph_vectors(self) -> ShapedRun:
        if len(self.glyph_ids) != len(self.advances):
            raise ValueError("glyph_ids and advances length mismatch")
        if self.offsets and len(self.offsets) != len(self.glyph_ids):
            raise ValueError("offsets and glyph_ids length mismatch")
        return self


class LayoutPlan(DomainModel):
    region_id: UUID
    revision_id: Sha256
    font_ref: ArtifactRef
    direction: Literal["horizontal", "vertical"]
    shaped_runs: tuple[ShapedRun, ...]
    line_breaks: tuple[int, ...] = ()
    position: Point
    alpha_mask_ref: ArtifactRef
    constraint_scores: dict[str, float] = Field(default_factory=dict)


class StageRecord(DomainModel):
    stage: StageName
    status: StageStatus
    fingerprint: Sha256
    input_hashes: tuple[Sha256, ...] = ()
    output_hashes: tuple[Sha256, ...] = ()
    cache_hit: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    issues: tuple[Issue, ...] = ()


class EntityRecord(DomainModel):
    entity_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("attributes")
    @classmethod
    def attributes_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_object(value, field_name="attributes")


class PageDocument(DomainModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source: SourcePage
    region_identities: tuple[RegionIdentity, ...] = ()
    region_revisions: tuple[RegionRevision, ...] = ()
    ocr_records: tuple[OCRRecord, ...] = ()
    style_fingerprints: tuple[StyleFingerprint, ...] = ()
    translations: tuple[TranslationRecord, ...] = ()
    layout_plans: tuple[LayoutPlan, ...] = ()
    stages: tuple[StageRecord, ...] = ()
    issues: tuple[Issue, ...] = ()
    entities: tuple[EntityRecord, ...] = ()

    @model_validator(mode="after")
    def validate_references_and_geometry(self) -> PageDocument:
        identities = {identity.region_id: identity for identity in self.region_identities}
        revisions = {revision.revision_id: revision for revision in self.region_revisions}
        if len(identities) != len(self.region_identities):
            raise ValueError("duplicate region_id")
        if len(revisions) != len(self.region_revisions):
            raise ValueError("duplicate revision_id")
        for identity in self.region_identities:
            revision = revisions.get(identity.active_revision_id)
            if revision is None or revision.region_id != identity.region_id:
                raise ValueError("active revision must exist and belong to region")
        for revision in self.region_revisions:
            if revision.region_id not in identities:
                raise ValueError("revision references unknown region")
            if revision.bbox.right > self.source.width or revision.bbox.bottom > self.source.height:
                raise ValueError("revision bbox outside source page")
            polygons = ([revision.polygon] if revision.polygon is not None else []) + list(
                revision.line_polygons
            )
            for polygon in polygons:
                if any(
                    point.x < 0
                    or point.y < 0
                    or point.x > self.source.width
                    or point.y > self.source.height
                    for point in polygon.points
                ):
                    raise ValueError("revision polygon outside source page")
        for record in (
            *self.ocr_records,
            *self.style_fingerprints,
            *self.translations,
            *self.layout_plans,
        ):
            revision = revisions.get(record.revision_id)
            if record.region_id not in identities or revision is None:
                raise ValueError("record references unknown region revision")
            if revision.region_id != record.region_id:
                raise ValueError("record revision must belong to record region")
        for issue in self.issues:
            if issue.page_id is not None and issue.page_id != self.source.page_id:
                raise ValueError("issue references a different source page")
            if issue.region_id is not None and issue.region_id not in identities:
                raise ValueError("issue references unknown region")
        for translation in self.translations:
            for issue in translation.issues:
                if issue.page_id is not None and issue.page_id != self.source.page_id:
                    raise ValueError("translation issue references a different source page")
                if issue.region_id is not None and issue.region_id != translation.region_id:
                    raise ValueError("translation issue references a different region")
        return self
