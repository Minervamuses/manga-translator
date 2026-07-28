"""Deterministic source-tree fingerprints and v0.3.2 baseline verification."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MANIFEST_RELATIVE_PATH = Path("benchmarks/baseline/v0.3.2/manifest.json")
LEGACY_TEST_EXCLUDES = {"tests/test_baseline_manifest.py"}
FIXTURE_PATHS = (
    "validation_samples/v032_caption_3_layout.json",
    "validation_samples/v032_caption_4_layout.json",
    "validation_samples/v032_caption_5_layout.json",
    "validation_samples/v032_dash_0211_layout.json",
    "validation_samples/v032_overlap_0188_layout.json",
)
ASSET_PATHS = (
    "models/comictextdetector.pt",
    "fonts/Iansui-Regular.ttf",
    "fonts/NotoSansCJKtc-Regular.otf",
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "htmlcov",
        "input",
        "output",
        "venv",
    }
)
EXCLUDED_FILE_NAMES = frozenset({".coverage"})
EXCLUDED_SUFFIXES = frozenset(
    {".bak", ".log", ".pyc", ".pyo", ".swp", ".temp", ".tmp", ".whl", ".zip"}
)


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class TreeFingerprint:
    sha256: str
    files: tuple[FileFingerprint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "sha256-path-size-content-v1",
            "sha256": self.sha256,
            "file_count": len(self.files),
            "files": [entry.as_dict() for entry in self.files],
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(relative_path: Path) -> bool:
    relative = relative_path.as_posix()
    if relative == MANIFEST_RELATIVE_PATH.as_posix():
        return True
    if any(part in EXCLUDED_DIRECTORY_NAMES or part.endswith(".egg-info") for part in relative_path.parts):
        return True
    return (
        relative_path.name in EXCLUDED_FILE_NAMES
        or relative_path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def iter_controlled_files(root: Path) -> Iterable[Path]:
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in EXCLUDED_DIRECTORY_NAMES and not name.endswith(".egg-info")
        )
        base = Path(directory)
        for filename in sorted(filenames):
            path = base / filename
            relative = path.relative_to(root)
            if not _is_excluded(relative):
                yield path


def fingerprint_tree(root: Path) -> TreeFingerprint:
    root = root.resolve()
    entries: list[FileFingerprint] = []
    for path in iter_controlled_files(root):
        relative = path.relative_to(root).as_posix()
        entries.append(
            FileFingerprint(path=relative, size=path.stat().st_size, sha256=sha256_file(path))
        )
    entries.sort(key=lambda entry: entry.path)

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")
    return TreeFingerprint(sha256=digest.hexdigest(), files=tuple(entries))


def fingerprint_manifest_entries(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compare_tree(
    expected_files: Iterable[dict[str, Any]], current: TreeFingerprint
) -> dict[str, list[str]]:
    expected = {str(item["path"]): item for item in expected_files}
    actual = {item.path: item for item in current.files}
    return {
        "added": sorted(set(actual) - set(expected)),
        "missing": sorted(set(expected) - set(actual)),
        "changed": sorted(
            path
            for path in set(expected) & set(actual)
            if expected[path].get("size") != actual[path].size
            or expected[path].get("sha256") != actual[path].sha256
        ),
    }


def collect_test_names(root: Path, *, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    names: list[str] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                names.append(f"{relative}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                names.extend(
                    f"{relative}::{node.name}::{child.name}"
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name.startswith("test_")
                )
    return sorted(names)


def _fixture_summary(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{relative}: fixture root must be an array")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "groups": len(data),
    }


def validate_fixture_schema(root: Path, fixture_paths: Iterable[str] = FIXTURE_PATHS) -> int:
    total = 0
    for relative in fixture_paths:
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"{relative}: fixture root must be an array")
        seen: set[str] = set()
        for index, item in enumerate(data):
            location = f"{relative}[{index}]"
            if not isinstance(item, dict):
                raise TypeError(f"{location}: group must be an object")
            group_id = item.get("id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{location}: id must be a non-empty string")
            if group_id in seen:
                raise ValueError(f"{location}: duplicate id {group_id}")
            seen.add(group_id)
            bbox = item.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, int) for value in bbox)
                or bbox[2] <= 0
                or bbox[3] <= 0
            ):
                raise ValueError(f"{location}: bbox must be [x, y, width, height]")
            for field in ("source", "translation", "status"):
                if not isinstance(item.get(field), str) or not item[field]:
                    raise ValueError(f"{location}: {field} must be a non-empty string")
            if not isinstance(item.get("valid"), bool):
                raise TypeError(f"{location}: valid must be boolean")
            if not isinstance(item.get("layout"), dict):
                raise TypeError(f"{location}: layout must be an object")
        total += len(data)
    return total


def build_manifest(root: Path, archive_sha256: str) -> dict[str, Any]:
    root = root.resolve()
    tree = fingerprint_tree(root)
    legacy_tests = collect_test_names(root, excluded=LEGACY_TEST_EXCLUDES)
    fixtures = [_fixture_summary(root, relative) for relative in FIXTURE_PATHS]
    assets = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size": (root / relative).stat().st_size,
        }
        for relative in ASSET_PATHS
    ]
    return {
        "schema_version": 1,
        "baseline_version": "0.3.2",
        "target_archive_sha256": archive_sha256.lower(),
        "source_tree": tree.as_dict(),
        "assets": assets,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "purpose": "historical_capture_metadata",
        },
        "runtime_contract": {
            "environment": {
                "manager": "conda",
                "definition": {
                    "path": "environment.yml",
                    "sha256": sha256_file(root / "environment.yml"),
                },
            },
            "python": {
                "implementation": "CPython",
                "major": 3,
                "minor": 11,
            },
        },
        "tests": {
            "unit": {
                "command": "PYTHONPATH=src python -m pytest -q",
                "legacy_count": len(legacy_tests),
                "legacy_tests": legacy_tests,
            },
            "model_integration": {
                "command": "PYTHONPATH=src python -m pytest -q -m model_integration",
                "default": "opt_in",
            },
            "api_integration": {
                "command": "PYTHONPATH=src python -m pytest -q -m api_integration",
                "default": "opt_in",
            },
        },
        "layout_fixtures": fixtures,
        "layout_fixture_group_count": sum(item["groups"] for item in fixtures),
        "fingerprint_exclusions": {
            "directories": sorted(EXCLUDED_DIRECTORY_NAMES),
            "file_names": sorted(EXCLUDED_FILE_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
            "self": MANIFEST_RELATIVE_PATH.as_posix(),
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"unsupported baseline manifest: {path}")
    return data


def _verify_runtime_contract(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = manifest.get("runtime_contract")
    if not isinstance(contract, dict):
        expected_python = manifest.get("python", {})
        if expected_python.get("implementation") != platform.python_implementation():
            errors.append("Python implementation differs from baseline")
        if expected_python.get("version") != platform.python_version():
            errors.append(
                f"Python version differs: expected {expected_python.get('version')}, "
                f"got {platform.python_version()}"
            )
        return errors

    environment = contract.get("environment", {})
    if environment.get("manager") != "conda":
        errors.append("runtime environment manager must be conda")
    elif not os.environ.get("CONDA_PREFIX"):
        errors.append("active Conda environment required")
    if os.environ.get("VIRTUAL_ENV"):
        errors.append("virtualenv must not be active when Conda manages the runtime")

    definition = environment.get("definition", {})
    definition_path = root / str(definition.get("path", ""))
    if not definition_path.is_file():
        errors.append(f"missing runtime definition: {definition.get('path')}")
    elif sha256_file(definition_path) != definition.get("sha256"):
        errors.append(f"changed runtime definition: {definition.get('path')}")

    expected_python = contract.get("python", {})
    if expected_python.get("implementation") != platform.python_implementation():
        errors.append("Python implementation differs from runtime contract")
    expected_version = (expected_python.get("major"), expected_python.get("minor"))
    actual_version = sys.version_info[:2]
    if expected_version != actual_version:
        errors.append(
            "Python major/minor differs from runtime contract: "
            f"expected {expected_version[0]}.{expected_version[1]}, "
            f"got {actual_version[0]}.{actual_version[1]}"
        )
    return errors


def verify_regression_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Verify immutable baseline evidence while allowing committed source evolution."""

    errors: list[str] = []

    for section in ("assets", "layout_fixtures"):
        for item in manifest.get(section, []):
            path = root / item["path"]
            if not path.is_file():
                errors.append(f"missing {section} file: {item['path']}")
            elif path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
                errors.append(f"changed {section} file: {item['path']}")

    try:
        group_count = validate_fixture_schema(
            root, (item["path"] for item in manifest.get("layout_fixtures", []))
        )
        if group_count != manifest.get("layout_fixture_group_count"):
            errors.append(
                "layout fixture group count mismatch: "
                f"expected {manifest.get('layout_fixture_group_count')}, got {group_count}"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"layout fixture schema invalid: {error}")

    known_tests = set(collect_test_names(root))
    unit = manifest.get("tests", {}).get("unit", {})
    legacy_tests = unit.get("legacy_tests", [])
    missing_tests = sorted(set(legacy_tests) - known_tests)
    if len(legacy_tests) != unit.get("legacy_count") or missing_tests:
        errors.append(
            f"legacy unit test inventory mismatch: count={len(legacy_tests)}; "
            f"missing={','.join(missing_tests)}"
        )

    errors.extend(_verify_runtime_contract(root, manifest))
    return errors


def verify_snapshot_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Verify the exact historical P0-01 source snapshot and its evidence."""

    errors = verify_regression_manifest(root, manifest)
    current = fingerprint_tree(root)
    expected_tree = manifest.get("source_tree", {})
    if current.sha256 != expected_tree.get("sha256"):
        differences = compare_tree(expected_tree.get("files", []), current)
        details = "; ".join(
            f"{kind}={','.join(paths)}" for kind, paths in differences.items() if paths
        )
        errors.insert(0, f"source tree fingerprint mismatch: {details or 'unknown difference'}")
    return errors


def verify_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    mode: Literal["snapshot", "regression"] = "snapshot",
) -> list[str]:
    if mode == "snapshot":
        return verify_snapshot_manifest(root, manifest)
    if mode == "regression":
        return verify_regression_manifest(root, manifest)
    raise ValueError(f"unknown baseline verification mode: {mode}")


def run_baseline_verification(
    root: Path | None = None,
    *,
    mode: Literal["snapshot", "regression"] = "snapshot",
) -> int:
    root = (root or project_root()).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest = load_manifest(manifest_path)
    errors = verify_manifest(root, manifest, mode=mode)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="manga-baseline-") as temporary:
        temporary_root = Path(temporary)
        commands = (
            (
                [
                    sys.executable,
                    "-X",
                    f"pycache_prefix={temporary_root / 'pycache'}",
                    "-m",
                    "compileall",
                    "-q",
                    "src",
                    "tests",
                ],
                os.environ.copy(),
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(temporary_root / "pytest"),
                    "-q",
                ],
                {
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(root / "src"),
                },
            ),
        )
        for command, environment in commands:
            result = subprocess.run(command, cwd=root, check=False, env=environment)
            if result.returncode:
                return result.returncode

    unit = manifest["tests"]["unit"]
    print(
        f"Baseline {mode} verification passed: "
        f"{unit['legacy_count']}/{unit['legacy_count']} legacy tests present"
    )
    if mode == "snapshot":
        print(f"historical_tree={manifest['source_tree']['sha256']}")
    else:
        print("historical source-tree comparison: skipped (regression mode)")
    print("model_integration: not run (opt-in)")
    print("api_integration: not run (opt-in)")
    return 0
