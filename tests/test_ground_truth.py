from __future__ import annotations

import copy
import json
from pathlib import Path

from manga_translator.benchmark.cli import main
from manga_translator.benchmark.ground_truth import (
    PROFILE_NAME,
    prepare_profile,
    validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _prepared(tmp_path: Path) -> Path:
    destination = tmp_path / PROFILE_NAME
    prepare_profile(ROOT, destination=destination)
    return destination


def test_prepare_imports_five_pages_and_38_regions(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    pages = sorted((destination / "pages").glob("*.json"))
    data = [json.loads(path.read_text(encoding="utf-8")) for path in pages]
    regions = [region for page in data for region in page["regions"]]
    assert len(pages) == 5
    assert len(regions) == 38
    assert sum(region["verified_by"] is None for region in regions) == 36


def test_known_0188_regions_are_corrected_and_verified(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    by_bbox = {
        tuple(region["bbox"][name] for name in ("x", "y", "width", "height")): region
        for region in page["regions"]
    }
    assert "セシリー・キャンベル" in by_bbox[(579, 1061, 92, 260)]["source_text"]["raw"]
    assert by_bbox[(579, 1061, 92, 260)]["verified_by"] == "execution-spec"
    assert by_bbox[(406, 1111, 146, 212)]["source_text"]["raw"].startswith("男が酒場で")


def test_swap_mutation_is_rejected(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    targets = [
        region
        for region in page["regions"]
        if tuple(region["bbox"][name] for name in ("x", "y", "width", "height"))
        in {(579, 1061, 92, 260), (406, 1111, 146, 212)}
    ]
    targets[0]["source_text"], targets[1]["source_text"] = (
        targets[1]["source_text"],
        targets[0]["source_text"],
    )
    targets[0]["fixed_translation"], targets[1]["fixed_translation"] = (
        targets[1]["fixed_translation"],
        targets[0]["fixed_translation"],
    )
    page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
    report = validate_profile(ROOT, profile_dir=destination)
    assert not report.ok
    assert sum("known 0188 anchor" in error for error in report.errors) == 2


def test_overlay_and_contact_sheet_labels_match_region_keys(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    for page_path in (destination / "pages").glob("*.json"):
        page = json.loads(page_path.read_text(encoding="utf-8"))
        labels = json.loads(
            (destination / "labels" / f"{page_path.stem}.labels.json").read_text(
                encoding="utf-8"
            )
        )
        expected = [region["region_key"] for region in page["regions"]]
        assert labels["overlay_labels"] == expected
        assert labels["contact_sheet_labels"] == expected
        assert (destination / "overlays" / f"{page_path.stem}.overlay.jpg").is_file()
        assert (destination / "contact_sheets" / f"{page_path.stem}.contact.jpg").is_file()


def test_validate_allows_unverified_but_strict_mode_blocks(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    report = validate_profile(ROOT, profile_dir=destination)
    assert report.ok
    assert report.pages == 5
    assert report.regions == 38
    assert report.unverified == 36
    strict = validate_profile(ROOT, profile_dir=destination, require_verified=True)
    assert not strict.ok
    assert "36 regions await human verification" in strict.errors


def test_checked_in_profile_is_fully_user_verified() -> None:
    report = validate_profile(ROOT)
    strict = validate_profile(ROOT, require_verified=True)
    pages = (ROOT / "benchmarks" / PROFILE_NAME / "pages").glob("*.json")
    regions = [
        region
        for page in pages
        for region in json.loads(page.read_text(encoding="utf-8"))["regions"]
    ]

    assert report.ok
    assert report.pages == 5
    assert report.regions == 38
    assert report.unverified == 0
    assert report.errors == []
    assert report.warnings == []
    assert strict.ok
    assert strict.errors == []
    assert {region["verified_by"] for region in regions} == {"garyc"}
    assert {region["verified_at"] for region in regions} == {"2026-07-29"}


def test_cli_require_verified_returns_zero() -> None:
    assert main(["--root", str(ROOT), "validate", PROFILE_NAME, "--require-verified"]) == 0


def test_checked_in_schema_is_applied_fail_closed(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    original = json.loads(page_path.read_text(encoding="utf-8"))
    mutations = [
        (
            lambda page: page.__setitem__("unexpected", True),
            "schema: page: unexpected property unexpected",
        ),
        (
            lambda page: page["regions"][0].__setitem__("unexpected", True),
            "unexpected property unexpected",
        ),
        (
            lambda page: page["regions"][4].__setitem__("verified_at", "2026-7-8"),
            "invalid ISO date",
        ),
        (
            lambda page: page["regions"][0].__setitem__("source_crop_sha256", "not-a-hash"),
            "string does not match",
        ),
        (
            lambda page: page["regions"][0].__setitem__("page_sha256", "0" * 64),
            "region page_sha256 mismatch",
        ),
    ]

    for mutate, expected_error in mutations:
        page = copy.deepcopy(original)
        mutate(page)
        page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

        report = validate_profile(ROOT, profile_dir=destination)

        assert not report.ok
        assert any(expected_error in error for error in report.errors)


def test_polygon_and_verification_coupling_are_validated(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    page["regions"][0]["polygon"] = [[0, 0], [10, 0], [page["width"] + 1, 10]]
    page["regions"][0]["verified_at"] = "2026-07-28"
    page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate_profile(ROOT, profile_dir=destination)

    assert any("polygon[2] outside page" in error for error in report.errors)
    assert any("verified_at must be null without verified_by" in error for error in report.errors)


def test_malformed_polygons_are_rejected_by_schema(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    original = json.loads(page_path.read_text(encoding="utf-8"))
    mutations = [
        ([[0, 0], [1, 1]], "requires at least 3 items"),
        ([[0, 0, 1], [1, 1], [2, 2]], "unexpected tuple items"),
        ([[True, 0], [1, 1], [2, 2]], "expected type number"),
        ([[float("nan"), 0], [1, 1], [2, 2]], "number must be finite"),
    ]

    for polygon, expected_error in mutations:
        page = copy.deepcopy(original)
        page["regions"][0]["polygon"] = polygon
        page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

        report = validate_profile(ROOT, profile_dir=destination)

        assert any(expected_error in error for error in report.errors)


def test_provenance_must_bind_to_existing_fixture_group(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    region = page["regions"][0]
    region["region_key"] = f"{page['page_sha256'][:12]}:missing"
    region["legacy_group_id"] = "missing"
    region["provenance"]["source_group_id"] = "missing"
    region["provenance"]["source_path"] = "../outside.json"
    page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate_profile(ROOT, profile_dir=destination)

    assert any("source_path must match fixture_source" in error for error in report.errors)
    assert any("source group missing from fixture" in error for error in report.errors)
    assert any("project path escapes repository" in error for error in report.errors)


def test_fixture_import_text_is_bound_to_provenance(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    page["regions"][0]["source_text"] = {"raw": "別の文", "nfc": "別の文"}
    page["regions"][0]["provenance"]["method"] = "invented_override"
    page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate_profile(ROOT, profile_dir=destination)

    assert any("imported text does not match provenance fixture" in error for error in report.errors)
    assert any("provenance method mismatch" in error for error in report.errors)


def test_page_and_fixture_paths_cannot_escape_repository(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    original = json.loads(page_path.read_text(encoding="utf-8"))

    for field in ("source_image", "fixture_source"):
        page = copy.deepcopy(original)
        page[field] = "../outside.json"
        page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

        report = validate_profile(ROOT, profile_dir=destination)

        assert any("project path escapes repository" in error for error in report.errors)


def test_missing_fixture_group_and_malformed_values_fail_without_crashing(tmp_path: Path) -> None:
    destination = _prepared(tmp_path)
    page_path = destination / "pages" / "0188_ive_hwa002.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    page["width"] = "not-an-integer"
    page["regions"][0]["source_text"]["raw"] = 42
    page["regions"][0]["bbox"]["x"] = -(10**100)
    page["regions"].pop()
    page_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
    labels_path = destination / "labels" / "0188_ive_hwa002.labels.json"
    labels_path.write_text("[]", encoding="utf-8")

    report = validate_profile(ROOT, profile_dir=destination)

    assert not report.ok
    assert any("expected type integer" in error for error in report.errors)
    assert any("missing source text" in error for error in report.errors)
    assert any("invalid bbox" in error for error in report.errors)
    assert any("fixture group coverage mismatch" in error for error in report.errors)
    assert any("expected 38 regions, found 37" in error for error in report.errors)
    assert any("labels artifact must be an object" in error for error in report.errors)
