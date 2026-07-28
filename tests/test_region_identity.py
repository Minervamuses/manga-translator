from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest

from manga_translator.domain.ids import dhash_distance, page_id_from_bytes, revision_id_for
from manga_translator.domain.models import (
    ArtifactRef,
    BoundingBox,
    PageDocument,
    SourcePage,
)
from manga_translator.domain.reconcile import (
    ReconciliationConfig,
    RegionObservation,
    reconcile_regions,
    trace_ancestors,
)

PAGE_BYTES = b"unchanged original page bytes"
PAGE_ID = page_id_from_bytes(PAGE_BYTES)
DETECTOR = "d" * 64


def _ids() -> Iterator[UUID]:
    value = 1
    while True:
        yield UUID(int=value)
        value += 1


def _factory(iterator: Iterator[UUID]):
    return lambda: next(iterator)


def _observation(x: float, *, width: float = 20.0, dhash: int = 0) -> RegionObservation:
    return RegionObservation(
        bbox=BoundingBox(x=x, y=10.0, width=width, height=20.0),
        detector_score=0.9,
        source="ctd",
        raw_index=int(x),
        crop_dhash=dhash,
    )


def _document(result) -> PageDocument:
    artifact = ArtifactRef(sha256=PAGE_ID, media_type="image/png", size_bytes=len(PAGE_BYTES))
    return PageDocument(
        source=SourcePage(
            page_id=PAGE_ID,
            original_bytes_sha256=PAGE_ID,
            source_path="page.png",
            width=200,
            height=100,
            mode="RGB",
            original_artifact=artifact,
        ),
        region_identities=result.identities,
        region_revisions=result.revisions,
        issues=result.issues,
    )


def test_page_and_revision_ids_are_deterministic_but_region_ids_are_persistent() -> None:
    observation = _observation(10.0)
    expected = revision_id_for(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        bbox=observation.bbox,
    )
    ids = _ids()
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[observation],
        id_factory=_factory(ids),
    )
    second = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[observation],
        previous=_document(first),
        id_factory=_factory(ids),
    )

    assert PAGE_ID == "94caa72ccdf81617291366fdbafe2b0aee97344cc2c1b797635ad4841763a4de"
    assert first.revisions[0].revision_id == expected
    assert second.identities[0].region_id == first.identities[0].region_id


def test_small_detector_drift_reuses_region_identity() -> None:
    ids = _ids()
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(20.0, dhash=0b1010)],
        id_factory=_factory(ids),
    )
    second = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(21.0, dhash=0b1010)],
        previous=_document(first),
        previous_crop_dhashes={first.revisions[0].revision_id: 0b1010},
        id_factory=_factory(ids),
    )

    assert second.identities[0].region_id == first.identities[0].region_id
    assert second.revisions[0].revision_id != first.revisions[0].revision_id


def test_close_candidates_are_ambiguous_instead_of_guessing() -> None:
    ids = _ids()
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(10.0), _observation(30.0)],
        id_factory=_factory(ids),
    )
    ambiguous = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(20.0)],
        previous=_document(first),
        config=ReconciliationConfig(match_threshold=0.3, ambiguity_margin=0.1),
        id_factory=_factory(ids),
    )

    old_ids = {item.region_id for item in first.identities}
    identity = ambiguous.identities[0]
    assert identity.region_id not in old_ids
    assert set(identity.lineage.possible_predecessors) == old_ids
    assert ambiguous.issues[0].code.value == "ambiguous_identity"


def test_competing_observations_do_not_claim_the_same_identity() -> None:
    ids = _ids()
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(20.0)],
        id_factory=_factory(ids),
    )
    second = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(19.5), _observation(20.5)],
        previous=_document(first),
        config=ReconciliationConfig(lineage_overlap_threshold=0.95),
        id_factory=_factory(ids),
    )

    assert len({item.region_id for item in second.identities}) == 2
    assert all(item.region_id != first.identities[0].region_id for item in second.identities)
    assert all(item.lineage.possible_predecessors for item in second.identities)


def test_merge_and_split_lineage_preserve_ancestors() -> None:
    ids = _ids()
    roots = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(10.0), _observation(31.0)],
        id_factory=_factory(ids),
    )
    merged = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(10.0, width=41.0)],
        previous=_document(roots),
        id_factory=_factory(ids),
    )
    merged_identity = merged.identities[0]
    root_ids = {item.region_id for item in roots.identities}
    assert merged_identity.lineage.reason == "merge"
    assert set(merged_identity.lineage.parents) == root_ids
    assert set(merged_identity.lineage.supersedes) == root_ids

    history = tuple(roots.identities) + tuple(merged.identities)
    split_previous = _document(merged)
    split = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="f" * 64,
        observations=[_observation(10.0), _observation(31.0)],
        previous=split_previous,
        id_factory=_factory(ids),
    )
    assert all(item.lineage.reason == "split" for item in split.identities)
    assert all(item.lineage.parents == (merged_identity.region_id,) for item in split.identities)

    complete_history = history + tuple(split.identities)
    for child in split.identities:
        assert set(trace_ancestors(complete_history, child.region_id)) == root_ids | {
            merged_identity.region_id
        }


def test_request_local_ids_are_not_part_of_revision_identity() -> None:
    observation = _observation(10.0)
    first = revision_id_for(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        bbox=observation.bbox,
    )
    second = revision_id_for(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        bbox=observation.bbox,
    )
    assert first == second


def test_identifier_inputs_are_strictly_bounded() -> None:
    with pytest.raises(ValueError, match="page_id"):
        revision_id_for(
            page_id="not-a-hash",
            detector_fingerprint=DETECTOR,
            bbox=_observation(10.0).bbox,
        )
    with pytest.raises(ValueError, match="fit within bits"):
        dhash_distance(1 << 64, 0)
