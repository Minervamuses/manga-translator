from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manga_translator import detector as detector_module
from manga_translator.config import DetectionConfig, PostprocessConfig
from manga_translator.detector import _blocks_to_regions, detect_text_regions
from manga_translator.geometry import clipped_raster_bbox
from manga_translator.stages.detect import detection_geometry_output


class Block:
    def __init__(self, *, lines, angle=0.0, vertical=True) -> None:
        self.xyxy = [-2.8, 5.9, 30.9, 45.2]
        self.lines = lines
        self.angle = angle
        self.vertical = vertical
        self.font_size = 17.5
        self.prob = 0.88


def test_rotated_block_retains_float_geometry_and_legacy_raster_bbox(monkeypatch) -> None:
    line = [[3.25, 8.5], [25.5, 6.75], [27.0, 38.25], [5.0, 40.0]]
    block = Block(lines=[line], angle=7.25)
    raw = np.zeros((60, 80), dtype=np.uint8)
    raw[8:40, 3:27] = 255

    monkeypatch.setattr(detector_module, "_get_detector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        detector_module,
        "_run_detector_pass",
        lambda *_args, **_kwargs: (raw, raw, [block]),
    )
    result = detect_text_regions(
        np.zeros((60, 80, 3), dtype=np.uint8),
        DetectionConfig(device="cpu", input_size=1024),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
    )
    region = result.regions_post[0]

    assert region.bbox == (0, 5, 30, 40)
    assert region.page_bbox == (0.0, 5.9, 30.9, 39.300000000000004)
    assert region.line_polygons == (tuple(tuple(point) for point in line),)
    assert region.angle_degrees == 7.25
    assert region.font_size_hint == 17.5
    assert region.mask_source is not None
    assert region.mask_source.detector_pass == 0
    assert region.mask_source.source_region_id == "r0000"
    assert region.local_mask is not None and np.any(region.local_mask)
    assert result.groups[0].bbox == region.bbox
    assert result.groups[0].mask_sources[0] == region.mask_source


def test_invalid_line_polygons_become_typed_detector_issues() -> None:
    blocks = [
        Block(lines=[[[1, 1], [2, 2], [3, 3]]]),
        Block(lines=[[[2, 2], [20, 20], [2, 20], [20, 2]]]),
        Block(lines=[[[-1, 2], [10, 2], [10, 10], [2, 10]]]),
    ]
    issues = []
    regions = _blocks_to_regions(
        blocks,
        60,
        80,
        source="ctd",
        input_size=1024,
        issues=issues,
    )

    assert len(regions) == 3
    assert all(not region.line_polygons for region in regions)
    assert {issue.code for issue in issues} == {
        "detector_degenerate_polygon",
        "detector_self_intersecting_polygon",
        "detector_out_of_bounds_polygon",
    }


def test_detect_stage_geometry_artifact_has_no_embedded_mask_bytes(monkeypatch) -> None:
    line = [[3.0, 8.0], [25.0, 8.0], [25.0, 38.0], [3.0, 38.0]]
    block = Block(lines=[line])
    raw = np.zeros((60, 80), dtype=np.uint8)
    raw[8:38, 3:25] = 255
    monkeypatch.setattr(detector_module, "_get_detector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        detector_module,
        "_run_detector_pass",
        lambda *_args, **_kwargs: (raw, raw, [block]),
    )
    detection = detect_text_regions(
        np.zeros((60, 80, 3), dtype=np.uint8),
        DetectionConfig(device="cpu"),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
    )
    output = detection_geometry_output(detection)
    payload = json.loads(output.artifacts[0].data)

    assert payload["schema_version"] == "detector_geometry.v1"
    assert payload["regions"][0]["angle_degrees"] == 0.0
    assert payload["regions"][0]["line_polygons"]
    assert payload["regions"][0]["raster_bbox"] == [0, 5, 30, 40]
    assert "local_mask" not in payload["regions"][0]


def test_all_38_baseline_group_bboxes_are_unchanged_by_raster_projection() -> None:
    checked = 0
    for path in sorted(Path("benchmarks/regression_v032/pages").glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        for region in page["regions"]:
            bbox = region["bbox"]
            actual = clipped_raster_bbox(
                (
                    bbox["x"],
                    bbox["y"],
                    bbox["x"] + bbox["width"],
                    bbox["y"] + bbox["height"],
                ),
                page_width=page["width"],
                page_height=page["height"],
            )
            assert actual == (
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"],
            )
            checked += 1
    assert checked == 38
