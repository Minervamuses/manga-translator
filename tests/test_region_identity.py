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

    assert second.current_identities[0].region_id == first.current_identities[0].region_id
    assert second.current_revisions[0].revision_id != first.current_revisions[0].revision_id
    assert len(second.revisions) == 2


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

    old_ids = {item.region_id for item in first.current_identities}
    identity = ambiguous.current_identities[0]
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

    assert len({item.region_id for item in second.current_identities}) == 2
    assert all(
        item.region_id != first.current_identities[0].region_id
        for item in second.current_identities
    )
    assert all(item.lineage.possible_predecessors for item in second.current_identities)


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
    merged_identity = merged.current_identities[0]
    root_ids = {item.region_id for item in roots.current_identities}
    assert merged_identity.lineage.reason == "merge"
    assert set(merged_identity.lineage.parents) == root_ids
    assert set(merged_identity.lineage.supersedes) == root_ids

    split_previous = _document(merged)
    split = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="f" * 64,
        observations=[_observation(10.0), _observation(31.0)],
        previous=split_previous,
        id_factory=_factory(ids),
    )
    assert all(item.lineage.reason == "split" for item in split.current_identities)
    assert all(
        item.lineage.parents == (merged_identity.region_id,)
        for item in split.current_identities
    )

    for child in split.current_identities:
        assert set(trace_ancestors(split.identities, child.region_id)) == root_ids | {
            merged_identity.region_id
        }
    assert len(split.identities) == 5
    assert sum(item.is_active for item in split.identities) == 2
    assert len(split.revisions) == 5


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


def test_revision_identity_includes_pose_and_ordered_masks() -> None:
    observation = _observation(10.0)
    first_mask = ArtifactRef(sha256="1" * 64, media_type="image/png", size_bytes=1)
    second_mask = ArtifactRef(sha256="2" * 64, media_type="image/png", size_bytes=1)

    def identity(**changes: object) -> str:
        inputs = {
            "page_id": PAGE_ID,
            "detector_fingerprint": DETECTOR,
            "bbox": observation.bbox,
            "angle_degrees": 12.5,
            "orientation": "rotated",
            "mask_refs": (first_mask, second_mask),
        }
        inputs.update(changes)
        return revision_id_for(**inputs)

    baseline = identity()
    assert identity(angle_degrees=12.6) != baseline
    assert identity(orientation="vertical") != baseline
    assert identity(mask_refs=(second_mask, first_mask)) != baseline
    assert identity(mask_refs=(first_mask, second_mask, first_mask)) != baseline


def test_content_geometry_conflict_is_ambiguous_instead_of_swapping_ids() -> None:
    ids = _ids()
    all_bits = (1 << 64) - 1
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(10.0, dhash=0), _observation(40.0, dhash=all_bits)],
        id_factory=_factory(ids),
    )
    previous_hashes = {
        revision.revision_id: observation.crop_dhash
        for revision, observation in zip(
            first.current_revisions,
            (_observation(10.0, dhash=0), _observation(40.0, dhash=all_bits)),
        )
    }
    swapped = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(10.0, dhash=all_bits), _observation(40.0, dhash=0)],
        previous=_document(first),
        previous_crop_dhashes=previous_hashes,
        id_factory=_factory(ids),
    )

    old_ids = {item.region_id for item in first.current_identities}
    assert not old_ids.intersection(item.region_id for item in swapped.current_identities)
    assert all(
        set(item.lineage.possible_predecessors) == old_ids
        for item in swapped.current_identities
    )
    assert len(swapped.issues) == 2


def test_reconciliation_retains_multiple_revision_generations() -> None:
    ids = _ids()
    first = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(20.0, dhash=1)],
        id_factory=_factory(ids),
    )
    second = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="e" * 64,
        observations=[_observation(21.0, dhash=1)],
        previous=_document(first),
        previous_crop_dhashes={first.current_revision_ids[0]: 1},
        id_factory=_factory(ids),
    )
    third = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint="f" * 64,
        observations=[_observation(22.0, dhash=1)],
        previous=_document(second),
        previous_crop_dhashes={second.current_revision_ids[0]: 1},
        id_factory=_factory(ids),
    )

    assert len(third.identities) == 1
    assert third.identities[0].is_active
    assert third.current_region_ids == first.current_region_ids
    assert len(third.revisions) == 3
    assert third.current_revisions[0].bbox.x == 22.0


def test_inactive_ancestor_is_never_an_exact_match_candidate() -> None:
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
    repeated_root_revision = reconcile_regions(
        page_id=PAGE_ID,
        detector_fingerprint=DETECTOR,
        observations=[_observation(10.0)],
        previous=_document(merged),
        config=ReconciliationConfig(match_threshold=1.0),
        id_factory=_factory(ids),
    )

    inactive_root_ids = {item.region_id for item in roots.current_identities}
    assert repeated_root_revision.current_identities[0].region_id not in inactive_root_ids
    assert all(not item.is_active for item in repeated_root_revision.identities if item.region_id in inactive_root_ids)
