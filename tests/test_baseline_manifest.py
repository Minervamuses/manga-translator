from __future__ import annotations

import json
import os
import subprocess
import tomllib
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
    verify_manifest,
    verify_regression_manifest,
    verify_snapshot_manifest,
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


def test_historical_snapshot_and_current_regression_are_distinct() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)

    snapshot_errors = verify_snapshot_manifest(root, manifest)
    regression_errors = verify_regression_manifest(root, manifest)

    assert any(error.startswith("source tree fingerprint mismatch:") for error in snapshot_errors)
    assert regression_errors == []


def test_runtime_contract_uses_conda_python_311_not_capture_patch(monkeypatch) -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    assert manifest["python"]["version"] == "3.12.10"
    assert manifest["python"]["purpose"] == "historical_capture_metadata"
    assert manifest["runtime_contract"]["python"] == {
        "implementation": "CPython",
        "major": 3,
        "minor": 11,
    }
    assert os.environ.get("CONDA_PREFIX")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    assert verify_manifest(root, manifest, mode="regression") == []


def test_runtime_contract_matches_conda_and_project_python_constraints() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    environment = (root / "environment.yml").read_text(encoding="utf-8")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "  - python=3.11\n" in environment
    assert project["project"]["requires-python"] == ">=3.11"
    assert manifest["runtime_contract"]["python"] == {
        "implementation": "CPython",
        "major": 3,
        "minor": 11,
    }


def test_runtime_contract_rejects_non_conda_execution(monkeypatch) -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    errors = verify_manifest(root, manifest, mode="regression")

    assert "active Conda environment required" in errors


def test_legacy_manifest_falls_back_to_exact_captured_python() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)
    manifest.pop("runtime_contract")

    errors = verify_manifest(root, manifest, mode="regression")

    assert any(error.startswith("Python version differs: expected 3.12.10") for error in errors)


def test_unknown_verification_mode_is_rejected() -> None:
    root = project_root()
    manifest = load_manifest(root / MANIFEST_RELATIVE_PATH)

    try:
        verify_manifest(root, manifest, mode="unknown")  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == "unknown baseline verification mode: unknown"
    else:
        raise AssertionError("unknown verification mode must fail closed")


def test_release_archive_excludes_runtime_and_build_files() -> None:
    paths = [
        "src/manga_translator/__pycache__/pipeline.cpython-311.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".ruff_cache/0/cache",
        ".venv/pyvenv.cfg",
        "build/lib/module.py",
        "dist/package.whl",
        "input/page.png",
        "output/page.png",
        "custom-output/artifacts/translation-responses/raw.json",
    ]
    result = subprocess.run(
        ["git", "check-attr", "export-ignore", "--", *paths],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert len(result.stdout.splitlines()) == len(paths)
    assert all(line.endswith("export-ignore: set") for line in result.stdout.splitlines())


def test_provider_response_artifacts_are_ignored_at_custom_output_paths() -> None:
    path = "custom-output/artifacts/translation-responses/raw.json"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", path],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == path
