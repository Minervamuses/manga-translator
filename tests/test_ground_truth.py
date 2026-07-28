from __future__ import annotations

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


def test_cli_require_verified_returns_nonzero() -> None:
    assert main(["--root", str(ROOT), "validate", PROFILE_NAME, "--require-verified"]) == 1
