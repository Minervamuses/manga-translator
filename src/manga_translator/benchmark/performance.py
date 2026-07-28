"""Reproducible v0.3.2 performance baseline generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import locale
import os
import platform
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ..config import AppConfig
from ..profiling import (
    RunProfiler,
    activate_profiler,
    measure_profiler_overhead,
    profile_page,
    profile_span,
    set_page_profile_metrics,
)

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.relative_to(root).as_posix(), "status": "missing", "sha256": None}
    return {
        "path": path.relative_to(root).as_posix(),
        "status": "present",
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _package_versions() -> dict[str, str | None]:
    names = ("manga-translator", "torch", "numpy", "opencv-python-headless", "httpx")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def _environment() -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "driver_version": None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(driver_version):
            cuda["driver_version"] = driver_version()
        cuda["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "packages": _package_versions(),
        "cuda": cuda,
        "power_profile": _power_profile(),
    }


def _load_corpus(root: Path) -> tuple[list[dict[str, Any]], str]:
    pages: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted((root / "benchmarks" / "regression_v032" / "pages").glob("*.json")):
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
        page = json.loads(raw)
        pages.append(
            {
                "fixture": path.name,
                "page_id": page["page_id"],
                "source_image": page["source_image"],
                "width": page["width"],
                "height": page["height"],
                "groups": len(page["regions"]),
                "ground_truth_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return pages, digest.hexdigest()


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
        with profile_span("mock_sample", sample_kind=sample_kind, repeat=repeat):
            for stage in ("decode", "detector_pass", "detector_postprocess"):
                with profile_span(stage, mock=True):
                    seed = _mock_stage_work(seed, 3)
            for group_index in range(int(page["groups"])):
                with profile_span("ocr_view", mock=True, group_index=group_index):
                    seed = _mock_stage_work(seed, 1)
                with profile_span("ocr_forward", mock=True, group_index=group_index):
                    seed = _mock_stage_work(seed, 2)
            for stage in ("translation", "layout", "inpaint", "render", "encode"):
                with profile_span(stage, mock=True):
                    seed = _mock_stage_work(seed, 2)
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _run_mock_baseline(run_id: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    profiler = RunProfiler(run_id, environment_kind="mock")
    cold: list[dict[str, Any]] = []
    warm: list[dict[str, Any]] = []
    with activate_profiler(profiler):
        for page in pages:
            cold.append(
                {
                    "page_id": page["page_id"],
                    "wall_ms": _mock_page(page, sample_kind="cold", repeat=0),
                }
            )
        for page in pages:
            _mock_page(page, sample_kind="warmup", repeat=0)
        for repeat in range(5):
            for page in pages:
                warm.append(
                    {
                        "page_id": page["page_id"],
                        "repeat": repeat,
                        "wall_ms": _mock_page(page, sample_kind="warm", repeat=repeat),
                    }
                )
    warm_values = [sample["wall_ms"] for sample in warm]
    worst = max(warm, key=lambda sample: sample["wall_ms"])
    return {
        "environment_kind": "mock",
        "status": "complete",
        "warmup_runs": 1,
        "repeats": 5,
        "cold": cold,
        "warm": warm,
        "summary": {
            "p50_wall_ms": statistics.median(warm_values),
            "p95_wall_ms": _percentile(warm_values, 0.95),
            "worst_page": worst,
        },
        "profiler": profiler.finish(),
    }


def _real_run_blockers(root: Path, pages: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if not torch.cuda.is_available():
        blockers.append("target_cuda_unavailable")
    try:
        config = AppConfig.from_yaml(root / "config.yaml")
        if not config.openrouter.api_key.strip() or config.openrouter.api_key == "YOUR_OPENROUTER_API_KEY":
            blockers.append("translation_api_key_unavailable")
    except Exception as error:  # noqa: BLE001 - benchmark records configuration blocker
        blockers.append(f"config_unavailable:{type(error).__name__}")
    for required in (
        root / "models" / "comictextdetector.pt",
        root / "fonts" / "Iansui-Regular.ttf",
    ):
        if not required.is_file():
            blockers.append(f"artifact_missing:{required.relative_to(root).as_posix()}")
    missing_pages = [page["source_image"] for page in pages if not (root / page["source_image"]).is_file()]
    if missing_pages:
        blockers.append(f"corpus_pages_missing:{len(missing_pages)}")
    return blockers or ["real_run_requires_explicit_cost_and_target_gpu_authorization"]


def run_performance_baseline(root: Path, profile: str = PROFILE_NAME) -> tuple[Path, dict[str, Any]]:
    if profile != PROFILE_NAME:
        raise ValueError(f"unknown performance profile: {profile}")
    root = root.resolve()
    pages, corpus_hash = _load_corpus(root)
    if len(pages) != 5:
        raise ValueError(f"v032_baseline requires 5 pages; found {len(pages)}")

    created = datetime.now(UTC)
    run_id = f"{profile}-{created.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_dir = root / "benchmarks" / "performance" / profile
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "config": _artifact(root / "config.yaml", root),
        "model": _artifact(root / "models" / "comictextdetector.pt", root),
        "font": _artifact(root / "fonts" / "Iansui-Regular.ttf", root),
        "fallback_font": _artifact(root / "fonts" / "NotoSansCJKtc-Regular.otf", root),
    }
    blockers = _real_run_blockers(root, pages)
    report = {
        "schema_version": "performance_baseline.v1",
        "profile": profile,
        "run_id": run_id,
        "created_at": created.isoformat(),
        "corpus": {"sha256": corpus_hash, "pages": pages},
        "environment": _environment(),
        "artifacts": artifacts,
        "profiler_overhead": measure_profiler_overhead(),
        "mock_run": _run_mock_baseline(run_id, pages),
        "real_run": {
            "environment_kind": "real",
            "status": "blocked",
            "blockers": blockers,
            "measurements": [],
        },
    }
    run_path = runs_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    relative_run = run_path.relative_to(root).as_posix()
    manifest_path = output_dir / "manifest.json"
    prior_runs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("schema_version") == "performance_manifest.v1":
                prior_runs = list(previous.get("runs", []))
        except (OSError, json.JSONDecodeError, TypeError):
            prior_runs = []
    manifest = {
        "schema_version": "performance_manifest.v1",
        "profile": profile,
        "latest_run_id": run_id,
        "runs": [
            *prior_runs,
            {
                "run_id": run_id,
                "path": relative_run,
                "sha256": _sha256_file(run_path),
                "environment_kind": "mock",
                "real_status": "blocked",
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_path, report
