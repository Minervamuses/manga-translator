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
TokenId = Annotated[int, Field(ge=0)]
LogProbability = Annotated[float, Field(le=0.0)]


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
    is_active: bool = True


class RasterTransform(DomainModel):
    source_space: str = Field(min_length=1)
    target_space: str = Field(min_length=1)
    affine_2x3: tuple[float, float, float, float, float, float]


class MaskLineage(DomainModel):
    artifact: ArtifactRef
    detector_pass: int
    detection_input_size: int = Field(ge=0)
    raw_index: int = Field(ge=-1)
    source_revision_id: Sha256 | None = None
    transform: RasterTransform


class RegionRevision(DomainModel):
    revision_id: Sha256
    region_id: UUID
    bbox: BoundingBox
    polygon: Polygon | None = None
    line_polygons: tuple[Polygon, ...] = ()
    angle_degrees: float = 0.0
    orientation: Literal["horizontal", "vertical", "rotated", "unknown"] = "unknown"
    kind: Literal["dialogue", "caption", "sfx", "other", "unknown"] = "unknown"
    detector_score: Score
    font_size_hint: float | None = Field(default=None, gt=0)
    mask_refs: tuple[ArtifactRef, ...] = ()
    mask_lineage: tuple[MaskLineage, ...] = ()
    source: str = Field(min_length=1)
    raw_index: int = Field(ge=-1)

    @model_validator(mode="after")
    def validate_mask_lineage(self) -> RegionRevision:
        mask_hashes = {artifact.sha256 for artifact in self.mask_refs}
        for lineage in self.mask_lineage:
            if lineage.artifact.sha256 not in mask_hashes:
                raise ValueError("mask lineage artifact must appear in mask_refs")
            if (
                lineage.source_revision_id is not None
                and lineage.source_revision_id != self.revision_id
            ):
                raise ValueError("region mask lineage must reference its owning revision")
        return self


class GroupGeometry(DomainModel):
    group_id: str = Field(min_length=1)
    member_revision_ids: tuple[Sha256, ...] = Field(min_length=1)
    bbox: BoundingBox
    polygon: Polygon | None = None
    union_mask_ref: ArtifactRef | None = None
    mask_lineage: tuple[MaskLineage, ...] = ()


class OCRCandidate(DomainModel):
    raw_text: str
    normalized_text: str
    token_scores: tuple[Score, ...] = ()
    confidence: Score
    confidence_kind: Literal["model", "calibrated", "heuristic", "ensemble", "unknown"]
    source_view: str = Field(min_length=1)
    token_ids: tuple[TokenId, ...] = ()
    token_logprobs: tuple[LogProbability, ...] = ()
    sequence: str = ""
    length_normalized_transition_logprob: LogProbability | None = None
    mean_token_entropy: float | None = Field(default=None, ge=0.0)
    mean_token_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    truncated: bool = False
    actual_batch_size: int = Field(default=1, ge=1)
    generation_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generation_config")
    @classmethod
    def validate_generation_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_object(value, field_name="generation_config")

    @model_validator(mode="after")
    def token_metrics_have_matching_lengths(self) -> OCRCandidate:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("token_ids and token_logprobs must have matching lengths")
        return self


class OCRRecord(DomainModel):
    region_id: UUID
    revision_id: Sha256
    candidates: tuple[OCRCandidate, ...]
    selected_index: int | None = Field(default=None, ge=0)
    model_id: str = Field(default="kha-white/manga-ocr-base", min_length=1)
    model_revision: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def selected_candidate_exists(self) -> OCRRecord:
        if self.selected_index is not None and self.selected_index >= len(self.candidates):
            raise ValueError("selected_index outside candidates")
        return self


class GroupOCRRecord(DomainModel):
    """One durable OCR result shared by every revision in a detected group."""

    group_id: str = Field(min_length=1)
    member_revision_ids: tuple[Sha256, ...] = Field(min_length=1)
    candidates: tuple[OCRCandidate, ...]
    selected_index: int | None = Field(default=None, ge=0)
    model_id: str = Field(default="kha-white/manga-ocr-base", min_length=1)
    model_revision: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def selected_candidate_exists(self) -> GroupOCRRecord:
        if self.selected_index is not None and self.selected_index >= len(self.candidates):
            raise ValueError("selected_index outside candidates")
        if len(set(self.member_revision_ids)) != len(self.member_revision_ids):
            raise ValueError("group OCR record has duplicate member revisions")
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
    sample_counts: dict[str, int] = Field(default_factory=dict)
    confidences: dict[str, Score] = Field(default_factory=dict)
    unknown_fields: tuple[str, ...] = ()
    shadow_offset: tuple[float, float] | None = None
    source: Literal["original_image"] = "original_image"


class TranslationRecord(DomainModel):
    region_id: UUID
    revision_id: Sha256
    request_item_id: str = Field(min_length=1)
    raw_response_ref: ArtifactRef
    validated_text: str
    entities: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()


class GroupTranslationRecord(DomainModel):
    """One provider result linked to all revisions in a detected group."""

    group_id: str = Field(min_length=1)
    member_revision_ids: tuple[Sha256, ...] = Field(min_length=1)
    request_item_id: str = Field(min_length=1)
    raw_response_ref: ArtifactRef
    validated_text: str
    entities: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()

    @model_validator(mode="after")
    def member_revisions_are_unique(self) -> GroupTranslationRecord:
        if len(set(self.member_revision_ids)) != len(self.member_revision_ids):
            raise ValueError("group translation record has duplicate member revisions")
        return self


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


class GroupLayoutRecord(DomainModel):
    """One durable layout-plan artifact linked to every revision in a group."""

    group_id: str = Field(min_length=1)
    member_revision_ids: tuple[Sha256, ...] = Field(min_length=1)
    plan_ref: ArtifactRef
    plan_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def member_revisions_are_unique(self) -> GroupLayoutRecord:
        if len(set(self.member_revision_ids)) != len(self.member_revision_ids):
            raise ValueError("group layout record has duplicate member revisions")
        return self


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


class PanelOverride(DomainModel):
    """Human-authored panel geometry and precedence for a page."""

    panel_id: str = Field(min_length=1)
    bbox: BoundingBox
    order: int = Field(ge=0)


class ReadingOrderOverride(DomainModel):
    """Human-authored final order for one persistent region identity."""

    region_id: UUID
    panel_id: str | None = Field(default=None, min_length=1)
    order: int = Field(ge=0)


class PageDocument(DomainModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source: SourcePage
    region_identities: tuple[RegionIdentity, ...] = ()
    region_revisions: tuple[RegionRevision, ...] = ()
    group_geometries: tuple[GroupGeometry, ...] = ()
    group_ocr_records: tuple[GroupOCRRecord, ...] = ()
    group_translations: tuple[GroupTranslationRecord, ...] = ()
    group_layout_records: tuple[GroupLayoutRecord, ...] = ()
    ocr_records: tuple[OCRRecord, ...] = ()
    style_fingerprints: tuple[StyleFingerprint, ...] = ()
    translations: tuple[TranslationRecord, ...] = ()
    layout_plans: tuple[LayoutPlan, ...] = ()
    stages: tuple[StageRecord, ...] = ()
    issues: tuple[Issue, ...] = ()
    entities: tuple[EntityRecord, ...] = ()
    panel_overrides: tuple[PanelOverride, ...] = ()
    reading_order_overrides: tuple[ReadingOrderOverride, ...] = ()

    def mapping_artifact_references(self) -> tuple[ArtifactRef, ...]:
        references: list[ArtifactRef] = []

        def visit(value: Any, *, entity_id: str) -> None:
            if isinstance(value, dict):
                artifact_keys = {"sha256", "media_type", "size_bytes"}
                if artifact_keys <= set(value):
                    try:
                        references.append(
                            ArtifactRef(
                                sha256=value["sha256"],
                                media_type=value["media_type"],
                                size_bytes=value["size_bytes"],
                            )
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"mapping entity {entity_id} has an invalid artifact reference"
                        ) from error
                for child in value.values():
                    visit(child, entity_id=entity_id)
            elif isinstance(value, list):
                for child in value:
                    visit(child, entity_id=entity_id)

        for entity in self.entities:
            if entity.kind != "mapping_snapshot":
                continue
            chain = entity.attributes.get("chain")
            if not isinstance(chain, dict):
                raise TypeError(f"mapping entity {entity.entity_id} has no mapping chain")
            visit(chain, entity_id=entity.entity_id)
        return tuple(references)

    @model_validator(mode="after")
    def validate_references_and_geometry(self) -> PageDocument:
        identities = {identity.region_id: identity for identity in self.region_identities}
        revisions = {revision.revision_id: revision for revision in self.region_revisions}
        if len(identities) != len(self.region_identities):
            raise ValueError("duplicate region_id")
        if len(revisions) != len(self.region_revisions):
            raise ValueError("duplicate revision_id")
        panel_ids = {panel.panel_id for panel in self.panel_overrides}
        if len(panel_ids) != len(self.panel_overrides):
            raise ValueError("duplicate panel override ID")
        if len({panel.order for panel in self.panel_overrides}) != len(self.panel_overrides):
            raise ValueError("duplicate panel override order")
        override_regions = {override.region_id for override in self.reading_order_overrides}
        if len(override_regions) != len(self.reading_order_overrides):
            raise ValueError("duplicate reading order override region")
        if len({override.order for override in self.reading_order_overrides}) != len(
            self.reading_order_overrides
        ):
            raise ValueError("duplicate reading order override position")
        for panel in self.panel_overrides:
            if panel.bbox.right > self.source.width or panel.bbox.bottom > self.source.height:
                raise ValueError("panel override bbox outside source page")
        for override in self.reading_order_overrides:
            if override.region_id not in identities:
                raise ValueError("reading order override references unknown region")
            if override.panel_id is not None and override.panel_id not in panel_ids:
                raise ValueError("reading order override references unknown panel")
        for identity in self.region_identities:
            revision = revisions.get(identity.active_revision_id)
            if revision is None or revision.region_id != identity.region_id:
                raise ValueError("active revision must exist and belong to region")
            lineage_refs = (
                *identity.lineage.parents,
                *identity.lineage.supersedes,
                *identity.lineage.possible_predecessors,
            )
            if identity.region_id in lineage_refs:
                raise ValueError("region lineage must not reference itself")
            for name, references in (
                ("parents", identity.lineage.parents),
                ("supersedes", identity.lineage.supersedes),
                ("possible_predecessors", identity.lineage.possible_predecessors),
            ):
                if len(set(references)) != len(references):
                    raise ValueError(f"region lineage {name} must not contain duplicates")
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
        for group in self.group_geometries:
            if any(revision_id not in revisions for revision_id in group.member_revision_ids):
                raise ValueError("group geometry references unknown revision")
            if len(set(group.member_revision_ids)) != len(group.member_revision_ids):
                raise ValueError("group geometry has duplicate member revisions")
            if group.bbox.right > self.source.width or group.bbox.bottom > self.source.height:
                raise ValueError("group geometry bbox outside source page")
            if group.union_mask_ref is None and group.mask_lineage:
                raise ValueError("group mask lineage requires a union mask artifact")
            if group.polygon is not None and any(
                point.x > self.source.width or point.y > self.source.height
                for point in group.polygon.points
            ):
                raise ValueError("group geometry polygon outside source page")
            for lineage in group.mask_lineage:
                if (
                    group.union_mask_ref is None
                    or lineage.artifact.sha256 != group.union_mask_ref.sha256
                ):
                    raise ValueError("group mask lineage must reference its union mask")
                if lineage.source_revision_id not in group.member_revision_ids:
                    raise ValueError("group mask lineage references a non-member revision")
        groups = {group.group_id: group for group in self.group_geometries}
        if len(groups) != len(self.group_geometries):
            raise ValueError("duplicate group geometry ID")
        for name, records in (
            ("OCR", self.group_ocr_records),
            ("translation", self.group_translations),
            ("layout", self.group_layout_records),
        ):
            if len({record.group_id for record in records}) != len(records):
                raise ValueError(f"duplicate group {name} record")
            for record in records:
                geometry = groups.get(record.group_id)
                if geometry is None:
                    raise ValueError(f"group {name} record references unknown group")
                if set(record.member_revision_ids) != set(geometry.member_revision_ids):
                    raise ValueError(
                        f"group {name} member revisions must match group geometry"
                    )
                if any(revision_id not in revisions for revision_id in record.member_revision_ids):
                    raise ValueError(f"group {name} record references unknown revision")
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
        for translation in self.group_translations:
            member_region_ids = {
                revisions[revision_id].region_id
                for revision_id in translation.member_revision_ids
            }
            for issue in translation.issues:
                if issue.page_id is not None and issue.page_id != self.source.page_id:
                    raise ValueError("translation issue references a different source page")
                if issue.region_id is not None and issue.region_id not in member_region_ids:
                    raise ValueError("translation issue references a different group")
        self.mapping_artifact_references()
        return self
