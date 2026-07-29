from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from manga_translator.domain.issues import Issue, IssueCode, IssueSeverity, StageName
from manga_translator.domain.models import (
    ArtifactRef,
    BoundingBox,
    EntityRecord,
    Lineage,
    OCRCandidate,
    OCRRecord,
    PageDocument,
    Point,
    Polygon,
    RegionIdentity,
    RegionRevision,
    SourcePage,
)
from manga_translator.domain.serialization import (
    DocumentSchemaError,
    MigrationRequired,
    UnsupportedSchemaVersion,
    canonical_document_bytes,
    migrate_document,
    parse_document,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REGION_ID = UUID("12345678-1234-5678-1234-567812345678")
OTHER_REGION_ID = UUID("87654321-4321-8765-4321-876543218765")


def _artifact(sha256: str = SHA_A) -> ArtifactRef:
    return ArtifactRef(sha256=sha256, media_type="image/png", size_bytes=123)


def _document() -> PageDocument:
    source = SourcePage(
        page_id=SHA_A,
        original_bytes_sha256=SHA_A,
        source_path="pages/page.png",
        width=100,
        height=200,
        mode="RGB",
        exif_orientation=1,
        original_artifact=_artifact(),
    )
    revision = RegionRevision(
        revision_id=SHA_B,
        region_id=REGION_ID,
        bbox=BoundingBox(x=10.0, y=20.0, width=30.0, height=40.0),
        polygon=Polygon(
            points=(
                Point(x=10.0, y=20.0),
                Point(x=40.0, y=20.0),
                Point(x=40.0, y=60.0),
                Point(x=10.0, y=60.0),
            )
        ),
        orientation="vertical",
        detector_score=0.9,
        mask_refs=(_artifact(SHA_B),),
        source="ctd",
        raw_index=0,
    )
    return PageDocument(
        source=source,
        region_identities=(
            RegionIdentity(
                region_id=REGION_ID,
                active_revision_id=SHA_B,
                lineage=Lineage(),
            ),
        ),
        region_revisions=(revision,),
        issues=(
            Issue(
                code=IssueCode.OCR_REJECTED,
                severity=IssueSeverity.WARNING,
                stage=StageName.OCR,
                page_id=SHA_A,
                region_id=REGION_ID,
            ),
        ),
    )


def test_canonical_round_trip_is_byte_identical() -> None:
    first = canonical_document_bytes(_document())
    parsed = parse_document(first)
    second = canonical_document_bytes(parsed)

    assert first == second
    assert first.endswith(b"\n")
    assert first.index(b'"issues"') < first.index(b'"schema_version"')


def test_identity_active_flag_defaults_for_pre_history_documents() -> None:
    payload = json.loads(canonical_document_bytes(_document()))
    del payload["region_identities"][0]["is_active"]

    parsed = parse_document(json.dumps(payload))

    assert parsed.region_identities[0].is_active is True


def test_newer_and_older_schemas_require_explicit_handling() -> None:
    payload = json.loads(canonical_document_bytes(_document()))
    payload["schema_version"] = "2.0"
    with pytest.raises(UnsupportedSchemaVersion):
        parse_document(json.dumps(payload))

    payload["schema_version"] = "0.9"
    with pytest.raises(MigrationRequired):
        parse_document(json.dumps(payload))
    migrated = migrate_document(payload)
    assert migrated.schema_version == "1.0"


def test_missing_required_field_and_unknown_field_are_rejected() -> None:
    payload = json.loads(canonical_document_bytes(_document()))
    del payload["source"]["source_path"]
    with pytest.raises(DocumentSchemaError, match="source_path"):
        parse_document(json.dumps(payload))

    with pytest.raises(ValidationError):
        ArtifactRef(
            sha256=SHA_A,
            media_type="image/png",
            size_bytes=10,
            raw_bytes=b"large payload",
        )


def test_invalid_geometry_and_out_of_page_geometry_are_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x=0.0, y=0.0, width=0.0, height=10.0)
    with pytest.raises(ValidationError, match="non-zero area"):
        Polygon(
            points=(
                Point(x=0.0, y=0.0),
                Point(x=1.0, y=1.0),
                Point(x=2.0, y=2.0),
            )
        )

    payload = json.loads(canonical_document_bytes(_document()))
    payload["region_revisions"][0]["bbox"]["x"] = 99.0
    with pytest.raises(DocumentSchemaError, match="outside source page"):
        parse_document(json.dumps(payload))

    with pytest.raises(ValidationError, match="vertices must be distinct"):
        Polygon(
            points=(
                Point(x=0.0, y=0.0),
                Point(x=2.0, y=0.0),
                Point(x=0.0, y=2.0),
                Point(x=0.0, y=0.0),
            )
        )
    with pytest.raises(ValidationError, match="must not self-intersect"):
        Polygon(
            points=(
                Point(x=0.0, y=0.0),
                Point(x=4.0, y=0.0),
                Point(x=0.0, y=3.0),
                Point(x=3.0, y=3.0),
            )
        )


def test_issue_code_is_typed_and_machine_readable() -> None:
    issue = _document().issues[0]
    assert issue.code is IssueCode.OCR_REJECTED
    assert issue.code.value == "ocr_rejected"
    with pytest.raises(ValidationError):
        Issue(
            code="human prose is not a code",
            severity=IssueSeverity.ERROR,
            stage=StageName.OCR,
        )


def test_non_finite_numbers_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Point(x=float("nan"), y=0.0)
    with pytest.raises(ValidationError):
        Point(x=float("inf"), y=0.0)


@pytest.mark.parametrize(
    "invalid",
    [
        {"payload": b"raw bytes"},
        {"nested": [1, {"value": float("nan")}]},
        {"not": ("a", "JSON array")},
        {1: "non-string key"},
    ],
)
def test_issue_details_and_entity_attributes_must_be_json_safe(invalid: object) -> None:
    with pytest.raises(ValidationError, match="JSON|keys must be strings|NaN|valid string"):
        Issue(
            code=IssueCode.OCR_REJECTED,
            severity=IssueSeverity.WARNING,
            stage=StageName.OCR,
            details=invalid,
        )
    with pytest.raises(ValidationError, match="JSON|keys must be strings|NaN|valid string"):
        EntityRecord(
            entity_id="entity",
            kind="test",
            canonical_name="entity",
            attributes=invalid,
        )


def test_records_must_reference_a_revision_owned_by_the_same_region() -> None:
    document = _document()
    other_revision = document.region_revisions[0].model_copy(
        update={"revision_id": SHA_C, "region_id": OTHER_REGION_ID}
    )
    candidate = OCRCandidate(
        raw_text="text",
        normalized_text="text",
        confidence=0.9,
        confidence_kind="model",
        source_view="original",
    )
    with pytest.raises(ValidationError, match="revision must belong"):
        PageDocument(
            source=document.source,
            region_identities=(
                *document.region_identities,
                RegionIdentity(region_id=OTHER_REGION_ID, active_revision_id=SHA_C),
            ),
            region_revisions=(*document.region_revisions, other_revision),
            ocr_records=(
                OCRRecord(
                    region_id=REGION_ID,
                    revision_id=SHA_C,
                    candidates=(candidate,),
                    selected_index=0,
                    model_revision="model",
                    preprocess_version="preprocess",
                ),
            ),
        )


def test_ocr_candidate_requires_consistent_json_safe_token_metrics() -> None:
    values = {
        "raw_text": "text",
        "normalized_text": "text",
        "confidence": 0.9,
        "confidence_kind": "model",
        "source_view": "original",
    }

    with pytest.raises(ValidationError, match="matching lengths"):
        OCRCandidate(**values, token_ids=(1, 2), token_logprobs=(-0.1,))
    with pytest.raises(ValidationError, match="NaN"):
        OCRCandidate(**values, generation_config={"temperature": float("nan")})

    candidate = OCRCandidate(
        **values,
        token_ids=(1, 2),
        token_logprobs=(-0.1, 0.0),
        length_normalized_transition_logprob=-0.05,
        generation_config={"max_length": 80, "do_sample": False},
    )
    assert candidate.token_ids == (1, 2)


def test_issue_scope_must_belong_to_the_document() -> None:
    document = _document()
    with pytest.raises(ValidationError, match="different source page"):
        PageDocument(
            source=document.source,
            region_identities=document.region_identities,
            region_revisions=document.region_revisions,
            issues=(
                Issue(
                    code=IssueCode.OCR_REJECTED,
                    severity=IssueSeverity.WARNING,
                    stage=StageName.OCR,
                    page_id=SHA_C,
                ),
            ),
        )


def test_mapping_snapshot_rejects_invalid_nested_artifact_reference() -> None:
    document = _document()
    with pytest.raises(ValidationError, match="invalid artifact reference"):
        PageDocument(
            source=document.source,
            region_identities=document.region_identities,
            region_revisions=document.region_revisions,
            entities=(
                EntityRecord(
                    entity_id="mapping-1",
                    kind="mapping_snapshot",
                    canonical_name="ready",
                    attributes={
                        "chain": {
                            "layout_plan": {
                                "artifact": {
                                    "sha256": "not-a-sha256",
                                    "media_type": "application/json",
                                    "size_bytes": 10,
                                }
                            }
                        }
                    },
                ),
            ),
        )
