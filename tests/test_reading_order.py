from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
import pytest

from manga_translator.domain.models import (
    ArtifactRef,
    BoundingBox,
    OCRCandidate,
    OCRRecord,
    PageDocument,
    PanelOverride,
    ReadingOrderOverride,
    RegionIdentity,
    RegionRevision,
    SourcePage,
)
from manga_translator.domain.serialization import canonical_document_bytes, parse_document
from manga_translator.order.panels import PanelCandidate, detect_panel_candidates
from manga_translator.reading_order import (
    OrderRegion,
    order_precedence_graph,
    resolve_reading_order,
)
from manga_translator.translation.units import build_translation_units

SOURCE_SHA = "a" * 64


def _id(number: int) -> UUID:
    return UUID(int=number)


def _revision_hash(number: int) -> str:
    return f"{number:064x}"


def _panel(name: str, x: float, y: float, width: float, height: float) -> PanelCandidate:
    return PanelCandidate(name, x, y, width, height, 0.9, "border")


def _order_region(
    number: int,
    box: tuple[float, float, float, float],
    orientation: str = "vertical",
) -> OrderRegion:
    return OrderRegion(
        _id(number), BoundingBox(x=box[0], y=box[1], width=box[2], height=box[3]), orientation
    )  # type: ignore[arg-type]


def _document(
    specs: list[tuple[int, tuple[float, float, float, float], str, str]],
    *,
    panel_overrides: tuple[PanelOverride, ...] = (),
    order_overrides: tuple[ReadingOrderOverride, ...] = (),
) -> PageDocument:
    artifact = ArtifactRef(sha256=SOURCE_SHA, media_type="image/png", size_bytes=1)
    identities = []
    revisions = []
    records = []
    for number, box, orientation, text in specs:
        region_id = _id(number)
        revision_id = _revision_hash(number)
        identities.append(RegionIdentity(region_id=region_id, active_revision_id=revision_id))
        revisions.append(
            RegionRevision(
                revision_id=revision_id,
                region_id=region_id,
                bbox=BoundingBox(x=box[0], y=box[1], width=box[2], height=box[3]),
                orientation=orientation,
                detector_score=0.9,
                source="fixture",
                raw_index=number,
            )
        )
        records.append(
            OCRRecord(
                region_id=region_id,
                revision_id=revision_id,
                candidates=(
                    OCRCandidate(
                        raw_text=text,
                        normalized_text=text,
                        confidence=0.8,
                        confidence_kind="calibrated",
                        source_view="raw",
                    ),
                ),
                selected_index=0,
                model_revision="fixture",
                preprocess_version="fixture",
            )
        )
    return PageDocument(
        source=SourcePage(
            page_id=SOURCE_SHA,
            original_bytes_sha256=SOURCE_SHA,
            source_path="fixture.png",
            width=200,
            height=200,
            mode="RGB",
            original_artifact=artifact,
        ),
        region_identities=tuple(identities),
        region_revisions=tuple(revisions),
        ocr_records=tuple(records),
        panel_overrides=panel_overrides,
        reading_order_overrides=order_overrides,
    )


def test_two_by_two_panels_are_top_first_then_right_to_left() -> None:
    panels = (
        _panel("top-left", 0, 0, 90, 90),
        _panel("top-right", 110, 0, 90, 90),
        _panel("bottom-left", 0, 110, 90, 90),
        _panel("bottom-right", 110, 110, 90, 90),
    )
    regions = (
        _order_region(1, (20, 20, 20, 40)),
        _order_region(2, (140, 20, 20, 40)),
        _order_region(3, (20, 130, 20, 40)),
        _order_region(4, (140, 130, 20, 40)),
    )

    result = resolve_reading_order(regions, panels=panels)

    assert [item.region_id for item in result.regions] == [_id(2), _id(1), _id(4), _id(3)]
    assert result.confidence == 0.9
    assert not result.order_uncertain


def test_cross_panel_region_falls_back_and_marks_uncertainty() -> None:
    panels = (_panel("left", 0, 0, 95, 200), _panel("right", 105, 0, 95, 200))
    regions = (_order_region(1, (70, 50, 60, 30)), _order_region(2, (150, 10, 20, 30)))

    result = resolve_reading_order(regions, panels=panels)

    assert result.order_uncertain
    assert result.confidence == 0.0
    assert result.issues[0].code == "order_uncertain"


def test_mixed_orientation_has_precedence_and_explicit_cycle_is_reported() -> None:
    vertical = _order_region(1, (120, 20, 20, 100), "vertical")
    horizontal = _order_region(2, (20, 50, 80, 20), "horizontal")
    result = resolve_reading_order((horizontal, vertical), panels=(_panel("only", 0, 0, 200, 200),))
    assert [item.region_id for item in result.regions] == [_id(1), _id(2)]

    ordered, cyclic = order_precedence_graph(
        (vertical, horizontal), ((_id(1), _id(2)), (_id(2), _id(1)))
    )
    assert cyclic
    assert [item.region_id for item in ordered] == [_id(1), _id(2)]


def test_recursive_xy_cut_finds_unframed_panel_layout() -> None:
    image = np.full((240, 240), 255, dtype=np.uint8)
    for left, top in ((10, 10), (140, 10), (10, 140), (140, 140)):
        cv2.rectangle(image, (left, top), (left + 89, top + 89), 180, thickness=-1)

    panels = detect_panel_candidates(image)

    xy_panels = [panel for panel in panels if panel.source == "xy_cut"]
    assert len(xy_panels) == 4
    assert all(panel.confidence >= 0.55 for panel in xy_panels)


def test_manual_override_round_trips_and_completely_replaces_auto_order() -> None:
    panel = PanelOverride(
        panel_id="manual", bbox=BoundingBox(x=0, y=0, width=200, height=200), order=0
    )
    overrides = (
        ReadingOrderOverride(region_id=_id(1), panel_id="manual", order=1),
        ReadingOrderOverride(region_id=_id(2), panel_id="manual", order=0),
    )
    document = _document(
        [(1, (150, 20, 20, 40), "vertical", "右"), (2, (20, 20, 20, 40), "vertical", "左")],
        panel_overrides=(panel,),
        order_overrides=overrides,
    )

    parsed = parse_document(canonical_document_bytes(document))
    built = build_translation_units(parsed, panels=(_panel("auto", 0, 0, 200, 200),))

    assert built.used_manual_override
    assert not built.order_uncertain
    assert [(unit.region_id, unit.request_item_id) for unit in built.units] == [
        (_id(2), "u0001"),
        (_id(1), "u0002"),
    ]


def test_sort_change_never_swaps_ocr_content_between_persistent_regions() -> None:
    document = _document(
        [(1, (150, 20, 20, 40), "vertical", "g004"), (2, (20, 20, 20, 40), "vertical", "g005")]
    )
    first = build_translation_units(document, panels=(_panel("page", 0, 0, 200, 200),))
    overrides = (
        ReadingOrderOverride(region_id=_id(1), panel_id="manual", order=1),
        ReadingOrderOverride(region_id=_id(2), panel_id="manual", order=0),
    )
    manual_panel = PanelOverride(
        panel_id="manual", bbox=BoundingBox(x=0, y=0, width=200, height=200), order=0
    )
    changed = document.model_copy(
        update={"panel_overrides": (manual_panel,), "reading_order_overrides": overrides}
    )
    second = build_translation_units(changed)

    assert (
        {unit.region_id: unit.ocr_raw for unit in first.units}
        == {unit.region_id: unit.ocr_raw for unit in second.units}
        == {_id(1): "g004", _id(2): "g005"}
    )
    assert [unit.region_id for unit in first.units] != [unit.region_id for unit in second.units]


def test_translation_units_use_only_active_revision_ocr_and_preserve_raw_nfc() -> None:
    document = _document(
        [
            (1, (150, 20, 20, 40), "vertical", "active"),
            (2, (20, 20, 20, 40), "vertical", "retired"),
        ]
    )
    active_record = document.ocr_records[0].model_copy(
        update={
            "candidates": (
                document.ocr_records[0].candidates[0].model_copy(
                    update={"raw_text": "カ\u3099", "normalized_text": "FILTERED"}
                ),
            )
        }
    )
    stale_revision = document.region_revisions[0].model_copy(
        update={"revision_id": "f" * 64, "raw_index": 99}
    )
    stale_record = document.ocr_records[0].model_copy(
        update={
            "revision_id": stale_revision.revision_id,
            "candidates": (
                document.ocr_records[0].candidates[0].model_copy(
                    update={"raw_text": "stale", "normalized_text": "stale"}
                ),
            ),
        }
    )
    manual_panel = PanelOverride(
        panel_id="manual", bbox=BoundingBox(x=0, y=0, width=200, height=200), order=0
    )
    revised = document.model_copy(
        update={
            "region_identities": (
                document.region_identities[0],
                document.region_identities[1].model_copy(update={"is_active": False}),
            ),
            "region_revisions": (*document.region_revisions, stale_revision),
            "ocr_records": (active_record, document.ocr_records[1], stale_record),
            "panel_overrides": (manual_panel,),
            "reading_order_overrides": (
                ReadingOrderOverride(region_id=_id(1), panel_id="manual", order=0),
                ReadingOrderOverride(region_id=_id(2), panel_id="manual", order=1),
            ),
        }
    )

    result = build_translation_units(
        PageDocument.model_validate(revised.model_dump(mode="python")),
        panels=(_panel("page", 0, 0, 200, 200),),
    )

    assert len(result.units) == 1
    assert result.units[0].region_id == _id(1)
    assert result.units[0].ocr_raw == "カ\u3099"
    assert result.units[0].ocr_nfc == "ガ"
    assert result.used_manual_override

    duplicated = revised.model_copy(
        update={"ocr_records": (active_record, active_record, document.ocr_records[1])}
    )
    with pytest.raises(ValueError, match="duplicate OCR record"):
        build_translation_units(duplicated)


def test_panel_candidates_reject_invalid_geometry_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="positive size"):
        _panel("invalid", 0, 0, 0, 100)
    with pytest.raises(ValueError, match="between zero and one"):
        PanelCandidate("invalid", 0, 0, 100, 100, 1.1, "border")

    regions = (_order_region(1, (20, 20, 20, 40)),)
    duplicate_panels = (
        _panel("same", 0, 0, 100, 200),
        _panel("same", 100, 0, 100, 200),
    )
    result = resolve_reading_order(regions, panels=duplicate_panels)

    assert result.order_uncertain
    assert "unique IDs" in result.issues[0].message
