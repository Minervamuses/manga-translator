from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manga_translator.benchmark.visual import (
    ARTIFACT_NAMES,
    VisualMetrics,
    bootstrap_visual_v1,
    build_review_sheet,
    build_visual_report,
    write_group_bundle,
)


def _metrics(index: int) -> VisualMetrics:
    return VisualMetrics(
        page_id=f"page-{index % 5}",
        region_key=f"group-{index:02d}",
        mapping_complete=True,
        missing_glyphs=0,
        clreq_hard_violations=0,
        accepted_collisions=0,
        outside_roi_changed_pixels=0,
        alpha_containment=0.995 + (index % 5) * 0.001,
        font_size_ratio=0.92 + (index % 4) * 0.02,
        center_offset_px=float(index % 7),
        whitespace_ratio=0.25 + (index % 3) * 0.03,
    )


def _reviews(count: int) -> list[dict]:
    return [
        {
            "region_key": f"group-{index:02d}",
            "reviewer": None,
            "decision": "tie",
            "critical_regression": False,
            "notes": None,
        }
        for index in range(count)
    ]


def test_visual_report_enforces_all_hard_metrics_and_reports_percentiles() -> None:
    report = build_visual_report((_metrics(index) for index in range(30)), reviews=_reviews(30))

    assert report["status"] == "passed"
    assert all(report["hard_metrics"].values())
    assert report["manual_review"]["verified_groups"] == 30
    for distribution in report["distributions"].values():
        assert distribution["p05"] <= distribution["p50"] <= distribution["p95"]
        assert distribution["worst"]["region_key"]


def test_visual_report_blocks_on_safety_metric_or_missing_human_review() -> None:
    records = [_metrics(index) for index in range(30)]
    records[4] = VisualMetrics(**{**records[4].__dict__, "outside_roi_changed_pixels": 1})

    report = build_visual_report(records)

    assert report["status"] == "blocked"
    assert report["automated_status"] == "failed"
    assert report["manual_review"]["status"] == "blocked"


def test_group_bundle_writes_all_eight_trace_artifacts(tmp_path: Path) -> None:
    image = np.full((24, 30, 3), 220, dtype=np.uint8)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[4:20, 6:24] = 255
    paths = write_group_bundle(
        tmp_path,
        source_overlay=image,
        safe_mask=mask,
        style_fingerprint={"source": "original_image"},
        shaped_runs=({"text": "中文", "direction": "ttb"},),
        layout_alpha=mask,
        inpainted_roi=image,
        final_preview=image,
        metrics=_metrics(0),
    )

    assert set(paths) == set(ARTIFACT_NAMES)
    assert all(Path(path).is_file() for path in paths.values())


def test_review_sheet_is_deterministic_and_uses_only_simple_decisions() -> None:
    groups = [
        {
            "page_id": "p0",
            "region_key": f"g{index}",
            "legacy_preview": f"legacy-{index}.png",
            "new_preview": f"new-{index}.png",
        }
        for index in range(30)
    ]
    sheet, sources = build_review_sheet(groups)
    repeated, _repeated_sources = build_review_sheet(reversed(groups))

    assert sheet == repeated
    assert len(sheet["rows"]) == 30
    assert sheet["allowed_decisions"] == ["new_better", "tie", "legacy_better"]
    assert all(row["decision"] is None for row in sheet["rows"])
    assert all(row["reviewer"] is None for row in sheet["rows"])
    assert len(sources["rows"]) == 30


def test_manual_gate_rejects_duplicate_or_unknown_review_regions() -> None:
    records = [_metrics(index) for index in range(30)]
    reviews = _reviews(30)
    duplicate = build_visual_report(records, reviews=[reviews[0]] * 30)
    unknown_rows = _reviews(30)
    unknown_rows[-1] = {**unknown_rows[-1], "region_key": "not-in-corpus"}
    unknown = build_visual_report(records, reviews=unknown_rows)

    assert duplicate["status"] == "blocked"
    assert duplicate["manual_review"]["duplicate_region_keys"] == ["group-00"]
    assert unknown["status"] == "blocked"
    assert unknown["manual_review"]["unknown_region_keys"] == ["not-in-corpus"]


def test_repository_visual_manifest_truthfully_records_current_blockers(tmp_path: Path) -> None:
    corpus = tmp_path / "benchmarks" / "regression_v032" / "pages"
    corpus.mkdir(parents=True)
    for page_index in range(5):
        page = {
            "page_id": f"p{page_index}",
            "source_image": f"source-{page_index}.png",
            "regions": [
                {"region_key": f"p{page_index}:g{group_index}", "verified_by": None}
                for group_index in range(6)
            ],
        }
        (corpus / f"p{page_index}.json").write_text(json.dumps(page), encoding="utf-8")

    manifest = bootstrap_visual_v1(tmp_path)

    assert manifest["status"] == "blocked"
    assert manifest["page_count"] == 5
    assert manifest["group_count"] == 30
    assert manifest["engine_switch"] == "not_performed"
    report = json.loads(
        (tmp_path / "benchmarks" / "visual_v1" / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "blocked"
    assert (tmp_path / "benchmarks" / "visual_v1" / "review_sheet.json").is_file()
