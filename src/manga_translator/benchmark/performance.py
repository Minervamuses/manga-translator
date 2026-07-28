"""Reproducible, non-authoritative v0.3.2 profiler smoke generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import locale
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import torch
import yaml

from ..manga_ocr_runtime import DEFAULT_MODEL_ID
from ..profiling import (
    RunProfiler,
    activate_profiler,
    measure_profiler_overhead,
    profile_page,
    profile_span,
    set_page_profile_metrics,
)
from .ground_truth import validate_profile

PROFILE_NAME = "v032_baseline"
REQUIRED_STAGES = (
    "decode",
    "detector_pass",
    "detector_postprocess",
    "ocr_view",
    "ocr_forward",
    "translation",
    "layout",
    "inpaint",
    "render",
    "encode",
)
SOURCE_FINGERPRINT_INPUTS = ("src", "scripts", "pyproject.toml", "environment.yml")
REQUIRED_CONDA_ENVIRONMENT = "manga"
REQUIRED_PYTHON = (3, 11)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _redact_channel(value: Any) -> str:
    raw = str(value or "")
    if "://" not in raw:
        return raw
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return "<invalid-url>"
    hostname = parsed.hostname
    if not hostname:
        return f"{parsed.scheme}://<redacted>"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        return f"{parsed.scheme}://{netloc}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    relative = _relative_path(path, root)
    if not path.is_file():
        return {
            "path": relative,
            "status": "missing",
            "sha256": None,
            "size_bytes": None,
        }
    return {
        "path": relative,
        "status": "present",
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        items,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).casefold(),
    )
    return {
        "algorithm": "canonical-json-sha256-v1",
        "sha256": _canonical_sha256(ordered),
        "count": len(ordered),
        "items": ordered,
    }


def _python_distribution_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, str]] = []
    for index, distribution in enumerate(importlib.metadata.distributions()):
        try:
            name = str(distribution.metadata.get("Name") or "").strip()
            version = str(distribution.version or "").strip()
            metadata = distribution.read_text("METADATA") or distribution.read_text("PKG-INFO")
            record = distribution.read_text("RECORD")
            direct_url = distribution.read_text("direct_url.json")
        except (OSError, UnicodeError, ValueError) as error:
            metadata_errors.append(
                {"distribution_index": str(index), "error": type(error).__name__}
            )
            continue
        if name:
            records.append(
                {
                    "name": name,
                    "version": version,
                    "metadata_sha256": _text_sha256(metadata),
                    "record_sha256": _text_sha256(record),
                    "direct_url_sha256": _text_sha256(direct_url),
                }
            )
    inventory = _inventory(records)
    inventory["metadata_errors"] = metadata_errors
    inventory["state_sha256"] = _canonical_sha256(
        {"items": inventory["items"], "metadata_errors": metadata_errors}
    )
    inventory["status"] = "degraded" if metadata_errors else "complete"
    return inventory


def _conda_inventory() -> dict[str, Any]:
    prefix_value = os.getenv("CONDA_PREFIX", "").strip()
    if not prefix_value:
        return {
            "status": "inactive",
            "algorithm": "canonical-json-sha256-v1",
            "sha256": _canonical_sha256([]),
            "count": 0,
            "items": [],
        }
    records: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, str]] = []
    for path in sorted((Path(prefix_value) / "conda-meta").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            metadata_errors.append({"file": path.name, "error": type(error).__name__})
            continue
        if not isinstance(payload, dict):
            metadata_errors.append({"file": path.name, "error": "root_not_object"})
            continue
        records.append(
            {
                "name": str(payload.get("name", "")),
                "version": str(payload.get("version", "")),
                "build": str(payload.get("build", "")),
                "build_number": payload.get("build_number"),
                "channel": _redact_channel(payload.get("channel")),
                "subdir": str(payload.get("subdir", "")),
                "package_sha256": payload.get("sha256"),
                "package_md5": payload.get("md5"),
                "metadata_sha256": _sha256_file(path),
            }
        )
    inventory = _inventory(records)
    inventory["metadata_errors"] = metadata_errors
    inventory["state_sha256"] = _canonical_sha256(
        {"items": inventory["items"], "metadata_errors": metadata_errors}
    )
    inventory["status"] = "active" if records and not metadata_errors else "degraded"
    return inventory


def _management_policy(root: Path) -> dict[str, Any]:
    violations: list[str] = []
    environment_path = root / "environment.yml"
    try:
        environment_definition = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        environment_definition = None
        violations.append("environment_definition_unavailable")
    dependencies_value = (
        environment_definition.get("dependencies", [])
        if isinstance(environment_definition, dict)
        else []
    )
    if not isinstance(dependencies_value, list):
        dependencies: list[Any] = []
        violations.append("environment_definition_dependencies_invalid")
    else:
        dependencies = dependencies_value
    environment_name = (
        environment_definition.get("name") if isinstance(environment_definition, dict) else None
    )
    if environment_name != REQUIRED_CONDA_ENVIRONMENT:
        violations.append("environment_yml_name_mismatch")
    python_specs = [
        item.strip()
        for item in dependencies
        if isinstance(item, str) and item.strip().casefold().startswith("python")
    ]
    required_python_prefix = f"python={'.'.join(map(str, REQUIRED_PYTHON))}"
    if not any(spec.casefold().startswith(required_python_prefix) for spec in python_specs):
        violations.append("environment_yml_python_constraint_mismatch")
    pip_subsection = any(
        isinstance(item, dict) and "pip" in item for item in dependencies
    )
    pip_package = any(
        isinstance(item, str)
        and (
            item.strip().casefold() == "pip"
            or item.strip().casefold().startswith(
                ("pip=", "pip<", "pip>", "pip!", "pip~", "pip ")
            )
        )
        for item in dependencies
    )
    if pip_subsection:
        violations.append("environment_yml_contains_pip_subsection")
    if pip_package:
        violations.append("environment_yml_installs_pip")
    poetry_lock = root / "poetry.lock"
    poetry_lock_status = "missing"
    if not poetry_lock.is_file():
        violations.append("poetry_lock_missing")
    else:
        try:
            lock_payload = tomllib.loads(poetry_lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            lock_payload = None
        metadata = lock_payload.get("metadata") if isinstance(lock_payload, dict) else None
        packages = lock_payload.get("package") if isinstance(lock_payload, dict) else None
        valid_lock = (
            isinstance(metadata, dict)
            and isinstance(metadata.get("lock-version"), str)
            and isinstance(metadata.get("content-hash"), str)
            and isinstance(packages, list)
        )
        poetry_lock_status = "valid" if valid_lock else "invalid"
        if not valid_lock:
            violations.append("poetry_lock_invalid")
    try:
        poetry_version = importlib.metadata.version("poetry")
    except importlib.metadata.PackageNotFoundError:
        poetry_version = None
        violations.append("poetry_not_installed_in_conda_environment")
    return {
        "required": {
            "environment_manager": "conda",
            "python_package_manager": "poetry",
            "conda_environment": REQUIRED_CONDA_ENVIRONMENT,
            "python_major_minor": list(REQUIRED_PYTHON),
        },
        "observed": {
            "active_conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
            "environment_yml_name": environment_name,
            "environment_yml_python_specs": python_specs,
            "poetry_version": poetry_version,
            "poetry_lock_status": poetry_lock_status,
            "environment_yml_has_pip_subsection": pip_subsection,
            "environment_yml_installs_pip": pip_package,
        },
        "compliant": not violations,
        "violations": violations,
    }


def _dependencies(root: Path) -> dict[str, Any]:
    return {
        "management_policy": _management_policy(root),
        "active_conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "active_conda_prefix": os.getenv("CONDA_PREFIX"),
        "definitions": {
            "environment_yml": _artifact(root / "environment.yml", root),
            "pyproject_toml": _artifact(root / "pyproject.toml", root),
            "poetry_lock": _artifact(root / "poetry.lock", root),
            "conda_lock": _artifact(root / "conda-lock.yml", root),
        },
        "conda_inventory": _conda_inventory(),
        "python_distributions": _python_distribution_inventory(),
    }


def _power_profile() -> dict[str, str]:
    if os.name != "nt":
        return {"status": "unavailable", "value": "unsupported_os"}
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "value": type(error).__name__}
    raw = completed.stdout or completed.stderr
    value = ""
    for encoding in ("utf-8", locale.getpreferredencoding(False), "utf-16"):
        try:
            value = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if not value:
        value = raw.decode("utf-8", errors="replace")
    value = " ".join(value.split())
    return {
        "status": "recorded" if completed.returncode == 0 else "unavailable",
        "value": value or f"exit_{completed.returncode}",
    }


def _total_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * page_count


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) or None


def _environment() -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "driver_version": _nvidia_driver_version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "total_memory_bytes": _total_memory_bytes(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "cuda": cuda,
        "power_profile": _power_profile(),
    }


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_fingerprint(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    for relative in SOURCE_FINGERPRINT_INPUTS:
        candidate = root / relative
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix.lower() not in {".pyc", ".pyo", ".tmp"}
            )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(set(files))
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return {
        "algorithm": "sha256-path-size-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "inputs": list(SOURCE_FINGERPRINT_INPUTS),
        "exclusions": ["benchmarks/performance", "cache/build/runtime outputs"],
    }


def _source_provenance(root: Path) -> dict[str, Any]:
    status = _git_value(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "worktree_clean_at_start": status == "" if status is not None else None,
        "worktree_changes_at_start": status.splitlines() if status else [],
        "source_fingerprint": _source_fingerprint(root),
    }


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _benchmark_precondition_errors(
    source: dict[str, Any],
    dependencies: dict[str, Any],
    environment: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix_value = dependencies.get("active_conda_prefix")
    prefix = Path(str(prefix_value)).resolve() if prefix_value else None
    executable = Path(str(environment.get("python_executable", "")))
    if dependencies.get("active_conda_environment") != REQUIRED_CONDA_ENVIRONMENT:
        errors.append(f"active_conda_environment_must_be_{REQUIRED_CONDA_ENVIRONMENT}")
    if dependencies.get("conda_inventory", {}).get("status") != "active":
        errors.append("conda_inventory_not_active_or_complete")
    if dependencies.get("python_distributions", {}).get("status") != "complete":
        errors.append("python_distribution_inventory_incomplete")
    if prefix is None or not _path_is_within(executable, prefix):
        errors.append("python_executable_outside_conda_prefix")
    try:
        python_version = tuple(
            int(part) for part in str(environment.get("python", "")).split(".")[:2]
        )
    except ValueError:
        python_version = ()
    if python_version != REQUIRED_PYTHON:
        required = ".".join(map(str, REQUIRED_PYTHON))
        errors.append(f"python_major_minor_must_be_{required}")
    if source.get("git_commit") is None or source.get("git_tree") is None:
        errors.append("git_provenance_unavailable")
    if source.get("worktree_clean_at_start") is not True:
        errors.append("worktree_not_clean_at_start")
    return errors


def _redacted_config_artifact(root: Path) -> tuple[dict[str, Any], bool]:
    path = root / "config.yaml"
    if not path.is_file():
        return _artifact(path, root), False
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError) as error:
        return {
            "path": _relative_path(path, root),
            "status": "invalid",
            "error": type(error).__name__,
            "sha256": None,
            "size_bytes": None,
        }, False
    if not isinstance(payload, dict):
        return {
            "path": _relative_path(path, root),
            "status": "invalid",
            "error": "root_not_object",
            "sha256": None,
            "size_bytes": None,
        }, False
    openrouter = payload.get("openrouter")
    configured = False
    if isinstance(openrouter, dict):
        key = str(openrouter.get("api_key") or "").strip()
        configured = bool(key and key != "YOUR_OPENROUTER_API_KEY")
        openrouter["api_key"] = "<redacted>"
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "path": _relative_path(path, root),
        "status": "present_redacted",
        "algorithm": "canonical-redacted-json-sha256-v1",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "size_bytes": len(canonical),
        "redacted_fields": ["openrouter.api_key"],
    }, configured


def _ocr_asset() -> dict[str, Any]:
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.getenv("HF_HUB_CACHE", hf_home / "hub"))
    repository = hub / f"models--{DEFAULT_MODEL_ID.replace('/', '--')}"
    reference = repository / "refs" / "main"
    try:
        revision = reference.read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    snapshot = repository / "snapshots" / revision if revision else None
    if snapshot is None or not snapshot.is_dir():
        return {
            "model_id": DEFAULT_MODEL_ID,
            "requested_revision": None,
            "resolved_revision": revision or None,
            "status": "missing",
            "snapshot_fingerprint": None,
        }
    records = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        records.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "blob": resolved.name,
                "content_sha256": _sha256_file(path),
            }
        )
    return {
        "model_id": DEFAULT_MODEL_ID,
        "requested_revision": None,
        "resolved_revision": revision,
        "status": "present_unpinned",
        "snapshot_fingerprint": {
            "algorithm": "hf-relative-path-size-content-sha256-v1",
            "sha256": _canonical_sha256(records),
            "file_count": len(records),
        },
    }


def _load_corpus(root: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    root = root.resolve()
    validation = validate_profile(root, require_verified=False)
    if not validation.ok:
        message = "; ".join(validation.errors)
        raise ValueError(f"regression_v032 corpus validation failed: {message}")
    pages: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    page_dir = root / "benchmarks" / "regression_v032" / "pages"
    for fixture_path in sorted(page_dir.glob("*.json")):
        if not _path_is_within(fixture_path, root):
            raise ValueError(f"corpus fixture escapes repository: {fixture_path}")
        fixture_bytes = fixture_path.read_bytes()
        page = json.loads(fixture_bytes)
        if not isinstance(page, dict):
            raise TypeError(f"corpus fixture root must be an object: {fixture_path}")
        source_relative = page.get("source_image")
        if not isinstance(source_relative, str) or not source_relative:
            raise ValueError(f"corpus source path is invalid: {fixture_path}")
        relative_path = Path(source_relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"corpus source escapes repository: {source_relative}")
        source_path = (root / relative_path).resolve()
        if not _path_is_within(source_path, root):
            raise ValueError(f"corpus source escapes repository: {source_relative}")
        source = _artifact(source_path, root)
        fixture = _artifact(fixture_path, root)
        page_id = page.get("page_id")
        page_sha256 = page.get("page_sha256")
        if not isinstance(page_id, str) or page_id != page_sha256:
            raise ValueError(f"corpus page identity mismatch: {fixture_path}")
        if source["status"] != "present" or source["sha256"] != page_sha256:
            raise ValueError(f"corpus source hash mismatch: {source_relative}")
        if page_id in seen_page_ids:
            raise ValueError(f"duplicate corpus page identity: {page_id}")
        seen_page_ids.add(page_id)
        aggregate_records.extend([fixture, source])
        pages.append(
            {
                "fixture": fixture_path.name,
                "fixture_path": fixture["path"],
                "fixture_sha256": fixture["sha256"],
                "source_image": source_relative,
                "source_sha256": source["sha256"],
                "source_size_bytes": source["size_bytes"],
                "page_id": page_id,
                "page_sha256": page_sha256,
                "identity_matches_source": True,
                "width": page["width"],
                "height": page["height"],
                "groups": len(page["regions"]),
            }
        )
    aggregate = [
        {
            "path": record["path"],
            "status": record["status"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in aggregate_records
    ]
    validation_summary = {
        "status": "valid_with_unverified_debt" if validation.unverified else "valid",
        "pages": validation.pages,
        "regions": validation.regions,
        "unverified": validation.unverified,
        "warnings": list(validation.warnings),
    }
    return pages, _canonical_sha256(aggregate), validation_summary


def _mock_stage_work(seed: bytes, rounds: int) -> bytes:
    value = seed
    for _ in range(max(1, rounds)):
        value = hashlib.sha256(value).digest()
    return value


def _mock_page(
    page: dict[str, Any],
    *,
    sample_kind: str,
    repeat: int,
    sample_id: str,
) -> float:
    started_ns = time.perf_counter_ns()
    page_id = str(page["page_id"])
    seed = page_id.encode("ascii")
    with profile_page(page_id, str(page["source_image"])):
        set_page_profile_metrics(
            page_id,
            width=page["width"],
            height=page["height"],
            final_groups=page["groups"],
        )
        with profile_span(
            "mock_sample",
            sample_id=sample_id,
            sample_kind=sample_kind,
            repeat=repeat,
        ):
            for stage in ("decode", "detector_pass", "detector_postprocess"):
                with profile_span(stage, mock=True, sample_id=sample_id):
                    seed = _mock_stage_work(seed, 3)
            for group_index in range(int(page["groups"])):
                with profile_span(
                    "ocr_view",
                    mock=True,
                    sample_id=sample_id,
                    group_index=group_index,
                ):
                    seed = _mock_stage_work(seed, 1)
                with profile_span(
                    "ocr_forward",
                    mock=True,
                    sample_id=sample_id,
                    group_index=group_index,
                ):
                    seed = _mock_stage_work(seed, 2)
            for stage in ("translation", "layout", "inpaint", "render", "encode"):
                with profile_span(stage, mock=True, sample_id=sample_id):
                    seed = _mock_stage_work(seed, 2)
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[min(len(ordered), rank) - 1])


def _mock_sample(
    page: dict[str, Any],
    *,
    sample_kind: str,
    repeat: int,
) -> dict[str, Any]:
    sample_id = f"{sample_kind}:{repeat}:{page['page_id']}"
    return {
        "sample_id": sample_id,
        "page_id": page["page_id"],
        "sample_kind": sample_kind,
        "repeat": repeat,
        "wall_ms": _mock_page(
            page,
            sample_kind=sample_kind,
            repeat=repeat,
            sample_id=sample_id,
        ),
    }


def _run_mock_baseline(run_id: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    profiler = RunProfiler(run_id, environment_kind="mock")
    cold: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    warm: list[dict[str, Any]] = []
    with activate_profiler(profiler):
        cold.extend(_mock_sample(page, sample_kind="cold", repeat=0) for page in pages)
        warmup.extend(_mock_sample(page, sample_kind="warmup", repeat=0) for page in pages)
        for repeat in range(5):
            warm.extend(
                _mock_sample(page, sample_kind="warm", repeat=repeat) for page in pages
            )
    warm_values = [sample["wall_ms"] for sample in warm]
    worst = max(warm, key=lambda sample: sample["wall_ms"])
    return {
        "environment_kind": "mock",
        "measurement_kind": "instrumentation_smoke",
        "status": "complete",
        "authoritative": False,
        "performance_claim_allowed": False,
        "warmup_runs_per_page": 1,
        "warmup_runs": 1,
        "repeats_per_page": 5,
        "repeats": 5,
        "cold": cold,
        "warmup": warmup,
        "warm": warm,
        "instrumentation_summary": {
            "p50_wall_ms": statistics.median(warm_values),
            "p95_wall_ms": _percentile(warm_values, 0.95),
            "worst_page": worst,
        },
        "summary": {
            "p50_wall_ms": statistics.median(warm_values),
            "p95_wall_ms": _percentile(warm_values, 0.95),
            "worst_page": worst,
            "authoritative": False,
        },
        "profiler": profiler.finish(),
    }


def _real_run_blockers(
    root: Path,
    pages: list[dict[str, Any]],
    *,
    credentials_configured: bool,
    dependencies: dict[str, Any],
    ocr_asset: dict[str, Any],
) -> list[str]:
    blockers = ["real_runner_not_implemented", "api_authorization_missing"]
    if not torch.cuda.is_available():
        blockers.append("target_cuda_unavailable")
    if not credentials_configured:
        blockers.append("translation_api_credentials_unavailable")
    if ocr_asset["status"] == "missing":
        blockers.append("ocr_model_snapshot_missing")
    if ocr_asset["requested_revision"] is None:
        blockers.append("ocr_model_revision_unpinned")
    blockers.extend(
        f"dependency_policy:{violation}"
        for violation in dependencies["management_policy"]["violations"]
    )
    for required in (
        root / "models" / "comictextdetector.pt",
        root / "fonts" / "Iansui-Regular.ttf",
        root / "fonts" / "NotoSansCJKtc-Regular.otf",
    ):
        if not required.is_file():
            blockers.append(f"artifact_missing:{required.relative_to(root).as_posix()}")
    missing_pages = [page["source_image"] for page in pages if page["source_sha256"] is None]
    if missing_pages:
        blockers.append(f"corpus_pages_missing:{len(missing_pages)}")
    return blockers


def _protocol(pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "invocation": (
            "conda run --no-capture-output -n manga env PYTHONDONTWRITEBYTECODE=1 "
            "PYTHONPATH=src python -B -m manga_translator.benchmark --root . performance "
            "--profile v032_baseline"
        ),
        "measurement_kind": "mock_infrastructure",
        "cold_semantics": "first synthetic instrumentation invocation per page",
        "warmup_runs_per_page": 1,
        "warm_repeats_per_page": 5,
        "page_order": [page["page_id"] for page in pages],
        "percentiles": {
            "p50": "statistics.median",
            "p95": "nearest-rank on all warm samples",
        },
        "units": {"wall": "milliseconds", "memory": "bytes"},
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {type(error).__name__}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} root must be an object")
    return payload


def _load_manifest_history(
    root: Path,
    output_dir: Path,
    manifest_path: Path,
    profile: str,
) -> tuple[str | None, str | None, list[tuple[dict[str, Any], dict[str, Any]]]]:
    if not manifest_path.exists():
        return None, None, []
    if not manifest_path.is_file():
        raise ValueError("performance manifest path is not a file")
    manifest = _read_json_object(manifest_path, label="performance manifest")
    schema = manifest.get("schema_version")
    if schema not in {"performance_manifest.v1", "performance_manifest.v2"}:
        raise ValueError(f"unsupported performance manifest schema: {schema!r}")
    if manifest.get("profile") != profile:
        raise ValueError("performance manifest profile mismatch")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("performance manifest runs must be an array of objects")
    latest_run_id = manifest.get("latest_run_id")
    run_ids: set[str] = set()
    paths: set[str] = set()
    history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        run_id = entry.get("run_id")
        relative = entry.get("path")
        expected_sha256 = entry.get("sha256")
        if not all(isinstance(value, str) and value for value in (run_id, relative, expected_sha256)):
            raise ValueError("performance manifest entry is missing run_id/path/sha256")
        if run_id in run_ids or relative in paths:
            raise ValueError("performance manifest contains duplicate run identity")
        run_ids.add(run_id)
        paths.add(relative)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"historical run path escapes repository: {relative}")
        run_path = (root / relative_path).resolve()
        if not _path_is_within(run_path, output_dir) or not run_path.is_file():
            raise ValueError(f"historical run artifact is missing or outside profile: {relative}")
        observed_sha256 = _sha256_file(run_path)
        if observed_sha256 != expected_sha256.casefold():
            raise ValueError(f"historical run artifact hash mismatch: {relative}")
        run_report = _read_json_object(run_path, label=f"historical run {run_id}")
        if run_report.get("run_id") != run_id or run_report.get("profile") != profile:
            raise ValueError(f"historical run identity mismatch: {relative}")
        history.append((dict(entry), run_report))
    if history and latest_run_id not in run_ids:
        raise ValueError("performance manifest latest_run_id is not present in runs")
    if not history and latest_run_id is not None:
        raise ValueError("performance manifest has latest_run_id without runs")
    return str(schema), str(latest_run_id) if latest_run_id is not None else None, history


def _nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _history_stale_reasons(
    previous_manifest_schema: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if (
        previous_manifest_schema != "performance_manifest.v2"
        or previous.get("schema_version") != current.get("schema_version")
    ):
        reasons.append("benchmark_schema_upgraded")
    comparisons = (
        (("corpus", "sha256"), "corpus_provenance_unavailable", "corpus_changed"),
        (
            ("source", "source_fingerprint", "sha256"),
            "source_provenance_unavailable",
            "source_changed",
        ),
        (
            ("dependencies",),
            "dependency_provenance_unavailable",
            "dependency_environment_changed",
        ),
        (("hardware",), "hardware_provenance_unavailable", "hardware_changed"),
        (("artifacts",), "artifact_provenance_unavailable", "artifacts_changed"),
        (("protocol",), "protocol_provenance_unavailable", "protocol_changed"),
    )
    for keys, unavailable, changed in comparisons:
        previous_value = _nested(previous, *keys)
        current_value = _nested(current, *keys)
        if previous_value is None:
            reasons.append(unavailable)
        elif current_value is not None and _canonical_sha256(previous_value) != _canonical_sha256(
            current_value
        ):
            reasons.append(changed)
    return reasons


def _historical_entries(
    previous_manifest_schema: str | None,
    previous_latest_run_id: str | None,
    history: list[tuple[dict[str, Any], dict[str, Any]]],
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    if previous_manifest_schema is None:
        return []
    historical_entries: list[dict[str, Any]] = []
    for entry, previous in history:
        historical = dict(entry)
        reasons = _history_stale_reasons(previous_manifest_schema, previous, current)
        if reasons:
            historical["compatibility"] = "historical_stale"
            historical["stale_reasons"] = reasons
            historical.pop("supersession_reason", None)
        else:
            historical["compatibility"] = "historical_compatible"
            historical["supersession_reason"] = "newer_run_recorded"
            historical.pop("stale_reasons", None)
        if entry.get("run_id") == previous_latest_run_id:
            historical["superseded_by"] = current["run_id"]
        historical_entries.append(historical)
    return historical_entries


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_performance_baseline(
    root: Path,
    profile: str = PROFILE_NAME,
) -> tuple[Path, dict[str, Any]]:
    if profile != PROFILE_NAME:
        raise ValueError(f"unknown performance profile: {profile}")
    root = root.resolve()
    output_dir = root / "benchmarks" / "performance" / profile
    manifest_path = output_dir / "manifest.json"
    previous_schema, previous_latest, history = _load_manifest_history(
        root,
        output_dir,
        manifest_path,
        profile,
    )
    pages, corpus_hash, validation_summary = _load_corpus(root)
    if len(pages) != 5:
        raise ValueError(f"v032_baseline requires 5 pages; found {len(pages)}")

    source = _source_provenance(root)
    dependencies = _dependencies(root)
    environment = _environment()
    precondition_errors = _benchmark_precondition_errors(source, dependencies, environment)
    if precondition_errors:
        raise RuntimeError(
            "performance baseline preconditions failed: " + "; ".join(precondition_errors)
        )

    created = datetime.now(UTC)
    run_id = f"{profile}-{created.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    config_artifact, credentials_configured = _redacted_config_artifact(root)
    ocr_asset = _ocr_asset()
    artifacts = {
        "effective_config": config_artifact,
        "detector_model": _artifact(root / "models" / "comictextdetector.pt", root),
        "ocr_model": ocr_asset,
        "font": _artifact(root / "fonts" / "Iansui-Regular.ttf", root),
        "fallback_font": _artifact(root / "fonts" / "NotoSansCJKtc-Regular.otf", root),
    }
    blockers = _real_run_blockers(
        root,
        pages,
        credentials_configured=credentials_configured,
        dependencies=dependencies,
        ocr_asset=ocr_asset,
    )
    report = {
        "schema_version": "performance_baseline.v2",
        "profile": profile,
        "run_id": run_id,
        "created_at": created.isoformat(),
        "source": source,
        "dependencies": dependencies,
        "environment": environment,
        "hardware": environment,
        "artifacts": artifacts,
        "corpus": {
            "algorithm": "canonical fixture-and-source artifact records sha256 v1",
            "sha256": corpus_hash,
            "ground_truth_validation": validation_summary,
            "pages": pages,
        },
        "protocol": _protocol(pages),
        "truth": {
            "measurement_kind": "mock_infrastructure",
            "authoritative": False,
            "performance_claim_allowed": False,
            "components": {
                "gpu_probe": "available" if torch.cuda.is_available() else "unavailable",
                "dependency_policy": (
                    "compliant"
                    if dependencies["management_policy"]["compliant"]
                    else "blocked"
                ),
                "corpus_human_verification": (
                    "complete"
                    if validation_summary["unverified"] == 0
                    else f"pending:{validation_summary['unverified']}"
                ),
                "real_detector": "not_run",
                "real_ocr": "not_run",
                "translation_api": "not_run",
                "full_pipeline": "not_run",
            },
        },
        "profiler_overhead": measure_profiler_overhead(),
        "mock_run": _run_mock_baseline(run_id, pages),
        "real_run": {
            "environment_kind": "real",
            "measurement_kind": "full_pipeline",
            "status": "blocked",
            "authoritative": False,
            "performance_claim_allowed": False,
            "runner_status": "not_implemented",
            "blockers": blockers,
            "measurements": [],
            "summary": None,
        },
    }

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{run_id}.json"
    if run_path.exists():
        raise FileExistsError(f"performance run already exists: {run_path}")
    _atomic_write_json(run_path, report)
    relative_run = run_path.relative_to(root).as_posix()
    historical_runs = _historical_entries(
        previous_schema,
        previous_latest,
        history,
        report,
    )
    manifest = {
        "schema_version": "performance_manifest.v2",
        "profile": profile,
        "latest_run_id": run_id,
        "runs": [
            *historical_runs,
            {
                "run_id": run_id,
                "path": relative_run,
                "sha256": _sha256_file(run_path),
                "measurement_kind": "mock_infrastructure",
                "authoritative": False,
                "performance_claim_allowed": False,
                "real_status": "blocked",
                "dependency_policy_compliant": dependencies["management_policy"][
                    "compliant"
                ],
                "compatibility": "current",
            },
        ],
    }
    try:
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        run_path.unlink(missing_ok=True)
        raise
    return run_path, report
