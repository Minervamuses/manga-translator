from __future__ import annotations

import json
from pathlib import Path

from manga_translator.baseline import (
    ASSET_PATHS,
    FIXTURE_PATHS,
    LEGACY_TEST_EXCLUDES,
    MANIFEST_RELATIVE_PATH,
    collect_test_names,
    compare_tree,
    fingerprint_manifest_entries,
    fingerprint_tree,
    load_manifest,
    project_root,
    validate_fixture_schema,
)


def test_fingerprint_is_deterministic_and_ignores_runtime_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_bytes(b"print('stable')\n")
    first = fingerprint_tree(tmp_path)

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.pyc").write_bytes(b"runtime")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "page.png").write_bytes(b"user output")
    second = fingerprint_tree(tmp_path)
    assert first == second


def test_fingerprint_detects_controlled_file_change(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_bytes(b"before")
    before = fingerprint_tree(tmp_path)
    source.write_bytes(b"after")
    after = fingerprint_tree(tmp_path)
    assert before.sha256 != after.sha256


def test_compare_tree_reports_added_missing_and_changed(tmp_path: Path) -> None:
    (tmp_path / "changed.txt").write_bytes(b"before")
    (tmp_path / "missing.txt").write_bytes(b"missing")
    expected = fingerprint_tree(tmp_path)

    (tmp_path / "changed.txt").write_bytes(b"after")
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "added.txt").write_bytes(b"added")
    differences = compare_tree([item.as_dict() for item in expected.files], fingerprint_tree(tmp_path))
    assert differences == {
        "added": ["added.txt"],
        "missing": ["missing.txt"],
        "changed": ["changed.txt"],
    }


def test_checked_in_manifest_is_internally_consistent() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    tree = manifest["source_tree"]
    assert tree["file_count"] == len(tree["files"])
    assert tree["sha256"] == fingerprint_manifest_entries(tree["files"])


def test_manifest_locks_assets_and_layout_fixtures() -> None:
    root = project_root()
    manifest = json.loads((root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["assets"]} == set(ASSET_PATHS)
    assert {item["path"] for item in manifest["layout_fixtures"]} == set(FIXTURE_PATHS)
    assert manifest["layout_fixture_group_count"] == 38
    assert manifest["target_archive_sha256"] == (
        "862c0f475910b0c7f334cbe7d51bfe1abd101c1669e87bb1110e6288608a3aa5"
    )


def test_layout_fixture_schema_covers_all_38_groups() -> None:
    assert validate_fixture_schema(project_root()) == 38


def test_manifest_preserves_67_legacy_test_names() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    legacy = manifest["tests"]["unit"]["legacy_tests"]
    assert len(legacy) == 67
    current = collect_test_names(root, excluded=LEGACY_TEST_EXCLUDES)
    assert set(legacy) <= set(current)
