from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from manga_translator.domain.issues import Issue, IssueCode, IssueSeverity, StageName
from manga_translator.domain.models import (
    ArtifactRef,
    BoundingBox,
    Lineage,
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
REGION_ID = UUID("12345678-1234-5678-1234-567812345678")


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
