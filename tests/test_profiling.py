from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manga_translator import pipeline as pipeline_module
from manga_translator.benchmark import performance as performance_module
from manga_translator.benchmark.performance import REQUIRED_STAGES, run_performance_baseline
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.profiling import (
    RunProfiler,
    activate_profiler,
    measure_profiler_overhead,
    profile_page,
    profile_span,
    record_api_profile,
    set_page_profile_metrics,
)
from manga_translator.result import PageResult

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_profiler_does_not_change_output_or_record_spans() -> None:
    def operation() -> str:
        value = "source"
        with profile_span("stage"):
            value = value.upper()
        return value

    expected = operation()
    profiler = RunProfiler("disabled", enabled=False)
    with activate_profiler(profiler):
        actual = operation()

    assert actual == expected == "SOURCE"
    assert profiler.spans == []


def test_profile_records_page_stage_resources_and_api_usage() -> None:
    profiler = RunProfiler("run-1", environment_kind="mock")
    with activate_profiler(profiler), profile_page("page-1", "page.png"):
        set_page_profile_metrics("page-1", width=100, height=200, final_groups=3)
        with profile_span("decode", fixture=True):
            sum(range(20))
        record_api_profile(
            model="test/model",
            status_code=200,
            latency_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.001},
        )

    report = profiler.finish()

    assert report["run_id"] == "run-1"
    assert report["environment_kind"] == "mock"
    assert report["pages"]["page-1"]["width"] == 100
    assert {span["stage"] for span in report["spans"]} == {"decode", "page_wall"}
    assert report["api_usage"][0]["usage"]["cost"] == 0.001
    assert report["resources"]["cpu_rss_peak_bytes"] is None or report["resources"][
        "cpu_rss_peak_bytes"
    ] > 0


def test_nested_same_page_profile_context_records_one_page_wall() -> None:
    profiler = RunProfiler("nested-page", environment_kind="mock")

    with (
        activate_profiler(profiler),
        profile_page("page-1", "page.png"),
        profile_page("page-1", "page.png"),
        profile_span("decode"),
    ):
        pass

    page_spans = [span for span in profiler.spans if span.page_id == "page-1"]
    assert [span.stage for span in page_spans].count("page_wall") == 1
    assert page_spans[-1].stage == "page_wall"


def test_process_single_page_reuses_precomputed_page_id(tmp_path, monkeypatch) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"stable-page-bytes")
    page_id = hashlib.sha256(source.read_bytes()).hexdigest()
    config = AppConfig(
        openrouter=OpenRouterConfig(api_key="test", model="test/model"),
        paths=PathsConfig(output_dir=tmp_path / "output"),
    )

    def fail_rehash(_path):
        raise AssertionError("page content must not be hashed twice")

    def process_staged(*, image_path, source_bytes, **_kwargs):
        assert source_bytes == b"stable-page-bytes"
        return PageResult(
            page_id=page_id,
            source_path=image_path,
            status="succeeded",
        )

    monkeypatch.setattr(pipeline_module, "_page_id_for_path", fail_rehash)
    monkeypatch.setattr(pipeline_module, "process_single_page_staged", process_staged)

    result = pipeline_module.process_single_page(
        source,
        config,
        {},
        page_id=page_id,
        state_dir=tmp_path / "state",
    )

    assert result.page_id == page_id


def test_run_pipeline_profiles_encode_inside_single_page_wall(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "page.png"
    source.write_bytes(b"stable-page-bytes")
    page_id = hashlib.sha256(source.read_bytes()).hexdigest()
    config = AppConfig(
        openrouter=OpenRouterConfig(api_key="test", model="test/model"),
        paths=PathsConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            glossary=tmp_path / "missing-glossary.json",
            font=tmp_path / "missing-font.ttf",
            font_fallback=tmp_path / "missing-fallback.ttf",
        ),
    )

    def process_page(*, image_path, **_kwargs):
        with profile_page(page_id, str(image_path)), profile_span("decode"):
            return PageResult(
                page_id=page_id,
                source_path=image_path,
                status="succeeded",
                image=np.full((8, 8, 3), 127, dtype=np.uint8),
            )

    monkeypatch.setattr(pipeline_module, "process_single_page_staged", process_page)
    profiler = RunProfiler("pipeline-page", environment_kind="mock")

    with activate_profiler(profiler):
        result = pipeline_module.run_pipeline(config)

    assert result.status == "succeeded"
    page_spans = [span for span in profiler.spans if span.page_id == page_id]
    page_walls = [span for span in page_spans if span.stage == "page_wall"]
    encode = next(span for span in page_spans if span.stage == "encode")
    assert len(page_walls) == 1
    assert encode.sequence < page_walls[0].sequence
    assert page_walls[0].wall_ms >= encode.wall_ms


def test_gpu_span_uses_cuda_events_and_synchronizes(monkeypatch) -> None:
    events: list[FakeEvent] = []
    synchronized = 0

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.recorded = False
            events.append(self)

        def record(self):
            self.recorded = True

        def elapsed_time(self, _other):
            return 3.25

    def synchronize():
        nonlocal synchronized
        synchronized += 1

    monkeypatch.setattr("manga_translator.profiling.torch.cuda.is_available", lambda: False)
    profiler = RunProfiler("gpu-test")
    monkeypatch.setattr("manga_translator.profiling.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("manga_translator.profiling.torch.cuda.Event", FakeEvent)
    monkeypatch.setattr("manga_translator.profiling.torch.cuda.synchronize", synchronize)

    with profiler.span("detector_pass", gpu=True):
        pass

    assert len(events) == 2
    assert all(event.recorded for event in events)
    assert synchronized == 1
    assert profiler.spans[0].gpu_ms == 3.25


def test_profiler_overhead_is_measured_separately() -> None:
    overhead = measure_profiler_overhead(iterations=10)

    assert overhead["iterations"] == 10
    assert overhead["disabled_ns_per_span"] >= 0
    assert overhead["enabled_ns_per_span"] >= 0


def _benchmark_root(tmp_path: Path) -> Path:
    pages_dir = tmp_path / "benchmarks" / "regression_v032" / "pages"
    pages_dir.mkdir(parents=True)
    source_dir = tmp_path / "samples" / "before_fix"
    source_dir.mkdir(parents=True)
    for index in range(5):
        source = source_dir / f"page{index}.png"
        source.write_bytes(f"page-{index}".encode())
        page_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        payload = {
            "page_id": page_sha256,
            "page_sha256": page_sha256,
            "source_image": source.relative_to(tmp_path).as_posix(),
            "width": 100 + index,
            "height": 200 + index,
            "regions": [{"region_key": f"r{item}"} for item in range(index + 1)],
        }
        (pages_dir / f"page{index}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    (tmp_path / "config.yaml").write_text(
        "openrouter:\n  api_key: sk-super-secret-marker\n  model: test/model\n",
        encoding="utf-8",
    )
    (tmp_path / "environment.yml").write_text(
        "name: manga\ndependencies:\n  - python=3.11\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "manga-translator"\nversion = "0.3.2"\n',
        encoding="utf-8",
    )
    return tmp_path


def _allow_synthetic_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(
        performance_module,
        "validate_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            pages=5,
            regions=15,
            unverified=15,
            errors=[],
            warnings=["15 regions await human verification"],
        ),
    )
    monkeypatch.setattr(performance_module, "_benchmark_precondition_errors", lambda *_: [])
    monkeypatch.setattr(performance_module, "_nvidia_driver_version", lambda: None)
    monkeypatch.setattr(
        performance_module,
        "_ocr_asset",
        lambda **_kwargs: {
            "model_id": "test/ocr",
            "requested_revision": None,
            "resolved_revision": None,
            "status": "missing",
            "snapshot_fingerprint": None,
        },
    )


def test_redacted_config_accepts_environment_credential_without_persisting_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "sk-environment-secret-marker"
    (tmp_path / "config.yaml").write_text(
        "openrouter:\n  api_key: YOUR_OPENROUTER_API_KEY\n  model: test/model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    artifact, configured = performance_module._redacted_config_artifact(tmp_path)

    assert configured
    assert artifact["status"] == "present_redacted"
    assert secret not in json.dumps(artifact)


def test_real_sample_accepts_region_rejection_after_complete_encode() -> None:
    result = SimpleNamespace(image=np.zeros((2, 2, 3), dtype=np.uint8))
    profile = {"spans": [{"stage": stage} for stage in REQUIRED_STAGES]}

    assert performance_module._real_sample_completion_errors(result, profile) == []


def test_real_sample_rejects_missing_output_and_required_stage() -> None:
    result = SimpleNamespace(image=None)
    profile = {"spans": [{"stage": stage} for stage in REQUIRED_STAGES[:-1]]}

    assert performance_module._real_sample_completion_errors(result, profile) == [
        "encoded_result_missing",
        "required_stages_missing:encode",
    ]


def test_performance_baseline_separates_mock_from_blocked_real_run(
    tmp_path,
    monkeypatch,
) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    monkeypatch.setattr(
        performance_module,
        "_power_profile",
        lambda: {"status": "unavailable", "value": "test"},
    )

    run_path, report = run_performance_baseline(root)

    assert run_path.is_file()
    assert report["schema_version"] == "performance_baseline.v2"
    assert report["corpus"]["sha256"]
    assert len(report["corpus"]["pages"]) == 5
    assert all(page["identity_matches_source"] for page in report["corpus"]["pages"])
    assert report["corpus"]["ground_truth_validation"]["unverified"] == 15
    assert report["source"]["source_fingerprint"]["sha256"]
    policy = report["dependencies"]["management_policy"]
    assert policy["required"]["environment_manager"] == "conda"
    assert policy["required"]["python_package_manager"] == "poetry"
    assert policy["compliant"] is False
    assert "poetry_lock_missing" in policy["violations"]
    assert report["dependencies"]["definitions"]["environment_yml"]["status"] == "present"
    assert report["dependencies"]["definitions"]["pyproject_toml"]["status"] == "present"
    assert report["dependencies"]["definitions"]["poetry_lock"]["status"] == "missing"
    assert report["truth"]["authoritative"] is False
    assert report["truth"]["performance_claim_allowed"] is False
    assert report["truth"]["components"]["dependency_policy"] == "blocked"
    assert "conda run" in report["protocol"]["invocation"]
    assert report["mock_run"]["environment_kind"] == "mock"
    assert report["mock_run"]["measurement_kind"] == "instrumentation_smoke"
    assert report["mock_run"]["authoritative"] is False
    assert report["mock_run"]["performance_claim_allowed"] is False
    assert report["mock_run"]["status"] == "complete"
    assert report["mock_run"]["warmup_runs"] == 1
    assert report["mock_run"]["repeats"] == 5
    assert report["mock_run"]["summary"]["p95_wall_ms"] >= 0
    assert report["mock_run"]["summary"]["worst_page"]["page_id"]
    assert report["real_run"]["environment_kind"] == "real"
    assert report["real_run"]["status"] == "blocked"
    assert report["real_run"]["measurements"] == []
    assert report["real_run"]["summary"] is None
    assert report["real_run"]["runner_status"] == "implemented_not_run"
    assert "corpus_human_verification_pending:15" in report["real_run"]["blockers"]
    assert "dependency_policy:poetry_lock_missing" in report["real_run"]["blockers"]
    samples = [
        *report["mock_run"]["cold"],
        *report["mock_run"]["warmup"],
        *report["mock_run"]["warm"],
    ]
    assert len(report["mock_run"]["cold"]) == 5
    assert len(report["mock_run"]["warmup"]) == 5
    assert len(report["mock_run"]["warm"]) == 25
    assert len({sample["sample_id"] for sample in samples}) == 35
    stages = {span["stage"] for span in report["mock_run"]["profiler"]["spans"]}
    assert set(REQUIRED_STAGES) <= stages
    page_walls = [
        span
        for span in report["mock_run"]["profiler"]["spans"]
        if span["stage"] == "page_wall"
    ]
    encodes = [
        span
        for span in report["mock_run"]["profiler"]["spans"]
        if span["stage"] == "encode"
    ]
    assert len(page_walls) == len(encodes) == 35
    for page_id in {sample["page_id"] for sample in samples}:
        page_encodes = sorted(span["sequence"] for span in encodes if span["page_id"] == page_id)
        page_ends = sorted(span["sequence"] for span in page_walls if span["page_id"] == page_id)
        assert all(encode < page_end for encode, page_end in zip(page_encodes, page_ends, strict=True))

    serialized = run_path.read_text(encoding="utf-8")
    assert "sk-super-secret-marker" not in serialized

    manifest_path = root / "benchmarks" / "performance" / "v032_baseline" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["runs"][0]
    assert manifest["schema_version"] == "performance_manifest.v2"
    assert entry["run_id"] == report["run_id"]
    assert entry["sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest()
    assert entry["authoritative"] is False
    assert entry["performance_claim_allowed"] is False
    assert entry["dependency_policy_compliant"] is False


def test_real_runner_preflight_reports_missing_gpu_model_and_api(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(performance_module.torch.cuda, "is_available", lambda: False)
    blockers = performance_module._real_run_blockers(
        tmp_path,
        [{"source_image": "missing.png", "source_sha256": None}],
        credentials_configured=False,
        dependencies={"management_policy": {"violations": []}},
        ocr_asset={"status": "missing", "requested_revision": None},
    )

    assert "target_cuda_unavailable" in blockers
    assert "translation_api_credentials_unavailable" in blockers
    assert "ocr_model_snapshot_missing" in blockers
    assert "ocr_model_revision_unpinned" in blockers
    assert "artifact_missing:models/comictextdetector.pt" in blockers
    assert "corpus_pages_missing:1" in blockers


def test_runner_enforces_preconditions_before_writing(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    monkeypatch.setattr(
        performance_module,
        "_benchmark_precondition_errors",
        lambda *_args: ["sentinel_precondition_failure"],
    )

    with pytest.raises(RuntimeError, match="sentinel_precondition_failure"):
        run_performance_baseline(root)

    assert not (root / "benchmarks" / "performance").exists()


def test_authorized_real_runner_publishes_authoritative_measurement(
    tmp_path, monkeypatch
) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    monkeypatch.setattr(
        performance_module,
        "validate_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            pages=5,
            regions=15,
            unverified=0,
            errors=[],
            warnings=[],
        ),
    )
    monkeypatch.setattr(performance_module, "_real_run_blockers", lambda *_args, **_kwargs: [])
    fake_real = {
        "environment_kind": "real",
        "measurement_kind": "full_pipeline",
        "status": "passed",
        "authoritative": True,
        "performance_claim_allowed": True,
        "runner_status": "implemented",
        "blockers": [],
        "measurements": [{"sample_id": "real-1", "wall_ms": 1.0}],
        "summary": {"p50_wall_ms": 1.0, "p95_wall_ms": 1.0},
    }
    monkeypatch.setattr(performance_module, "_run_real_pipeline", lambda *_args: fake_real)

    run_path, report = run_performance_baseline(root, execute_real=True)

    assert run_path.is_file()
    assert report["real_run"] == fake_real
    assert report["truth"]["authoritative"] is True
    assert report["truth"]["performance_claim_allowed"] is True
    manifest = json.loads(
        (root / "benchmarks/performance/v032_baseline/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["runs"][-1]["real_status"] == "passed"
    assert manifest["runs"][-1]["authoritative"] is True


def test_real_runner_failure_does_not_publish_partial_run(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    monkeypatch.setattr(
        performance_module,
        "validate_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            pages=5,
            regions=15,
            unverified=0,
            errors=[],
            warnings=[],
        ),
    )
    monkeypatch.setattr(performance_module, "_real_run_blockers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        performance_module,
        "_run_real_pipeline",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("sample failed")),
    )

    with pytest.raises(RuntimeError, match="sample failed"):
        run_performance_baseline(root, execute_real=True)

    assert not (root / "benchmarks/performance/v032_baseline/manifest.json").exists()
    assert not (root / "benchmarks/performance/v032_baseline/runs").exists()


def test_atomic_json_failure_preserves_existing_destination(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"old-manifest")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(performance_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        performance_module._atomic_write_json(destination, {"new": True})

    assert destination.read_bytes() == b"old-manifest"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_manifest_write_failure_removes_new_run(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    atomic_write = performance_module._atomic_write_json

    def fail_manifest(path, payload):
        if path.name == "manifest.json":
            raise OSError("simulated manifest failure")
        atomic_write(path, payload)

    monkeypatch.setattr(performance_module, "_atomic_write_json", fail_manifest)

    with pytest.raises(OSError, match="simulated manifest failure"):
        run_performance_baseline(root)

    output = root / "benchmarks" / "performance" / "v032_baseline"
    assert list((output / "runs").glob("*.json")) == []
    assert not (output / "manifest.json").exists()


def test_corpus_hash_covers_fixture_and_source_bytes(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    _pages, original, _validation = performance_module._load_corpus(root)

    source = root / "samples" / "before_fix" / "page0.png"
    source.write_bytes(b"mutated-source")
    fixture = root / "benchmarks" / "regression_v032" / "pages" / "page0.json"
    with pytest.raises(ValueError, match="source hash mismatch"):
        performance_module._load_corpus(root)

    payload = json.loads(fixture.read_text(encoding="utf-8"))
    changed_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    payload["page_id"] = changed_sha256
    payload["page_sha256"] = changed_sha256
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    _pages, source_changed, _validation = performance_module._load_corpus(root)

    fixture.write_text(fixture.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _pages, fixture_changed, _validation = performance_module._load_corpus(root)

    assert original != source_changed
    assert source_changed != fixture_changed


def test_corpus_validation_fails_before_hashing_artifacts(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    monkeypatch.setattr(
        performance_module,
        "validate_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            errors=["source image escapes repository"],
        ),
    )
    monkeypatch.setattr(
        performance_module,
        "_artifact",
        lambda *_args: pytest.fail("invalid corpus must fail before artifact hashing"),
    )

    with pytest.raises(ValueError, match="corpus validation failed"):
        performance_module._load_corpus(root)


@pytest.mark.parametrize("source_image", ["../outside.png", "/tmp/outside.png"])
def test_corpus_source_path_cannot_escape_repository(
    tmp_path,
    monkeypatch,
    source_image,
) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    fixture = root / "benchmarks" / "regression_v032" / "pages" / "page0.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    payload["source_image"] = source_image
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source escapes repository"):
        performance_module._load_corpus(root)


def test_checked_in_corpus_is_valid_for_non_authoritative_smoke() -> None:
    pages, corpus_sha256, validation = performance_module._load_corpus(ROOT)

    assert len(pages) == 5
    assert sum(page["groups"] for page in pages) == 38
    assert len(corpus_sha256) == 64
    assert validation["status"] == "valid"
    assert validation["unverified"] == 0


def test_source_fingerprint_excludes_performance_outputs(tmp_path) -> None:
    root = _benchmark_root(tmp_path)
    source_file = root / "src" / "manga_translator" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    before = performance_module._source_fingerprint(root)

    generated = root / "benchmarks" / "performance" / "v032_baseline" / "runs" / "run.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("generated", encoding="utf-8")
    after_output = performance_module._source_fingerprint(root)
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    after_source = performance_module._source_fingerprint(root)

    assert before["sha256"] == after_output["sha256"]
    assert before["sha256"] != after_source["sha256"]


def test_source_provenance_treats_untracked_files_as_dirty(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git_value(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments[0] == "status":
            return "?? src/untracked.py"
        return "deadbeef"

    monkeypatch.setattr(performance_module, "_git_value", fake_git_value)

    provenance = performance_module._source_provenance(tmp_path)

    assert provenance["worktree_clean_at_start"] is False
    assert provenance["worktree_changes_at_start"] == ["?? src/untracked.py"]
    assert ("status", "--porcelain=v1", "--untracked-files=all") in calls


def test_benchmark_preconditions_reject_wrong_environment_and_dirty_tree() -> None:
    valid_source = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "worktree_clean_at_start": True,
    }
    valid_dependencies = {
        "active_conda_environment": "manga",
        "active_conda_prefix": "/opt/conda/envs/manga",
        "conda_inventory": {"status": "active"},
        "python_distributions": {"status": "complete"},
    }
    valid_environment = {
        "python": "3.11.15",
        "python_executable": "/opt/conda/envs/manga/bin/python",
    }

    assert (
        performance_module._benchmark_precondition_errors(
            valid_source,
            valid_dependencies,
            valid_environment,
        )
        == []
    )

    invalid_source = {**valid_source, "worktree_clean_at_start": False}
    invalid_dependencies = {
        **valid_dependencies,
        "active_conda_environment": "base",
        "conda_inventory": {"status": "inactive"},
        "python_distributions": {"status": "degraded"},
    }
    invalid_environment = {"python": "3.12.0", "python_executable": "/usr/bin/python"}
    errors = performance_module._benchmark_precondition_errors(
        invalid_source,
        invalid_dependencies,
        invalid_environment,
    )

    assert "active_conda_environment_must_be_manga" in errors
    assert "conda_inventory_not_active_or_complete" in errors
    assert "python_distribution_inventory_incomplete" in errors
    assert "python_executable_outside_conda_prefix" in errors
    assert "python_major_minor_must_be_3.11" in errors
    assert "worktree_not_clean_at_start" in errors


def test_active_benchmark_runtime_is_required_conda_environment() -> None:
    dependencies = performance_module._dependencies(ROOT)
    environment = performance_module._environment()
    source = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "worktree_clean_at_start": True,
    }

    errors = performance_module._benchmark_precondition_errors(
        source,
        dependencies,
        environment,
    )

    assert dependencies["active_conda_environment"] == "manga"
    assert dependencies["conda_inventory"]["status"] == "active"
    assert dependencies["python_distributions"]["status"] == "complete"
    assert errors == []


def test_management_policy_reports_missing_poetry_lock_and_pip_definition(
    tmp_path,
    monkeypatch,
) -> None:
    root = _benchmark_root(tmp_path)
    (root / "environment.yml").write_text(
        "name: wrong\ndependencies:\n  - python=3.12\n  - pip\n  - pip:\n      - package==1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(performance_module.importlib.metadata, "version", lambda _name: "2.2.1")

    policy = performance_module._management_policy(root)

    assert policy["compliant"] is False
    assert policy["observed"]["poetry_version"] == "2.2.1"
    assert policy["observed"]["poetry_lock_status"] == "missing"
    assert "poetry_lock_missing" in policy["violations"]
    assert "environment_yml_contains_pip_subsection" in policy["violations"]
    assert "environment_yml_installs_pip" in policy["violations"]
    assert "environment_yml_name_mismatch" in policy["violations"]
    assert "environment_yml_python_constraint_mismatch" in policy["violations"]


def test_management_policy_rejects_invalid_poetry_lock(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    (root / "poetry.lock").write_text("not valid toml =", encoding="utf-8")
    monkeypatch.setattr(performance_module.importlib.metadata, "version", lambda _name: "2.2.1")

    policy = performance_module._management_policy(root)

    assert policy["observed"]["poetry_lock_status"] == "invalid"
    assert "poetry_lock_invalid" in policy["violations"]
    assert "poetry_lock_missing" not in policy["violations"]


def test_conda_channel_redaction_removes_credentials_and_query() -> None:
    redacted = performance_module._redact_channel(
        "https://user:secret@example.invalid/private/channel?token=secret"
    )

    assert redacted == "https://example.invalid"
    assert "user" not in redacted
    assert "secret" not in redacted
    assert "private" not in redacted


def test_ocr_snapshot_fingerprint_hashes_file_contents(tmp_path, monkeypatch) -> None:
    hub = tmp_path / "hub"
    repository = hub / "models--kha-white--manga-ocr-base"
    snapshot = repository / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("revision-1\n", encoding="utf-8")
    model_file = snapshot / "model.bin"
    model_file.write_bytes(b"first")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    before = performance_module._ocr_asset()
    model_file.write_bytes(b"other")
    after = performance_module._ocr_asset()
    pinned = performance_module._ocr_asset(requested_revision="revision-1")

    assert before["resolved_revision"] == "revision-1"
    assert before["snapshot_fingerprint"]["algorithm"].endswith("content-sha256-v1")
    assert before["snapshot_fingerprint"]["sha256"] != after["snapshot_fingerprint"]["sha256"]
    assert pinned["requested_revision"] == "revision-1"
    assert pinned["resolved_revision"] == "revision-1"
    assert pinned["status"] == "present_pinned"


def test_inventory_and_percentile_algorithms_are_deterministic() -> None:
    first = performance_module._inventory(
        [
            {"name": "Beta", "version": "2", "build": "b"},
            {"name": "alpha", "version": "1", "build": "a"},
        ]
    )
    second = performance_module._inventory(list(reversed(first["items"])))

    assert first["sha256"] == second["sha256"]
    assert performance_module._percentile(list(range(1, 21)), 0.95) == 19.0


def test_v2_manifest_preserves_and_marks_v1_run_stale(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    output = root / "benchmarks" / "performance" / "v032_baseline"
    runs = output / "runs"
    runs.mkdir(parents=True)
    old_run = runs / "historical-v1.json"
    old_run.write_text(
        json.dumps(
            {
                "schema_version": "performance_baseline.v1",
                "profile": "v032_baseline",
                "run_id": "historical-v1",
                "corpus": {"sha256": "old-corpus"},
            }
        ),
        encoding="utf-8",
    )
    old_bytes = old_run.read_bytes()
    old_entry = {
        "run_id": "historical-v1",
        "path": "benchmarks/performance/v032_baseline/runs/historical-v1.json",
        "sha256": hashlib.sha256(old_bytes).hexdigest(),
        "environment_kind": "mock",
        "real_status": "blocked",
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "performance_manifest.v1",
                "profile": "v032_baseline",
                "latest_run_id": "historical-v1",
                "runs": [old_entry],
            }
        ),
        encoding="utf-8",
    )

    run_path, report = run_performance_baseline(root)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest_run_id"] == report["run_id"]
    assert manifest["runs"][0]["run_id"] == old_entry["run_id"]
    assert manifest["runs"][0]["sha256"] == old_entry["sha256"]
    assert manifest["runs"][0]["compatibility"] == "historical_stale"
    assert "benchmark_schema_upgraded" in manifest["runs"][0]["stale_reasons"]
    assert "corpus_changed" in manifest["runs"][0]["stale_reasons"]
    assert manifest["runs"][0]["superseded_by"] == report["run_id"]
    assert manifest["runs"][1]["sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest()
    assert old_run.read_bytes() == old_bytes


@pytest.mark.parametrize(
    "manifest_bytes",
    [b"{broken", b'{"schema_version":"unknown","profile":"v032_baseline","runs":[]}'],
)
def test_invalid_manifest_is_not_overwritten(tmp_path, monkeypatch, manifest_bytes) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    output = root / "benchmarks" / "performance" / "v032_baseline"
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    with pytest.raises(ValueError):
        run_performance_baseline(root)

    assert manifest_path.read_bytes() == manifest_bytes
    assert not (output / "runs").exists()


def test_manifest_rejects_tampered_historical_run(tmp_path, monkeypatch) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)
    output = root / "benchmarks" / "performance" / "v032_baseline"
    runs = output / "runs"
    runs.mkdir(parents=True)
    old_run = runs / "historical-v1.json"
    old_run.write_text(
        json.dumps(
            {
                "schema_version": "performance_baseline.v1",
                "profile": "v032_baseline",
                "run_id": "historical-v1",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "performance_manifest.v1",
        "profile": "v032_baseline",
        "latest_run_id": "historical-v1",
        "runs": [
            {
                "run_id": "historical-v1",
                "path": "benchmarks/performance/v032_baseline/runs/historical-v1.json",
                "sha256": "0" * 64,
            }
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="hash mismatch"):
        run_performance_baseline(root)

    assert manifest_path.read_bytes() == before


def test_v2_rerun_marks_same_provenance_as_historical_compatible(
    tmp_path,
    monkeypatch,
) -> None:
    root = _benchmark_root(tmp_path)
    _allow_synthetic_benchmark(monkeypatch)

    first_path, first_report = run_performance_baseline(root)
    first_bytes = first_path.read_bytes()
    _second_path, second_report = run_performance_baseline(root)

    manifest_path = root / "benchmarks" / "performance" / "v032_baseline" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_entry = next(
        entry for entry in manifest["runs"] if entry["run_id"] == first_report["run_id"]
    )
    assert first_entry["compatibility"] == "historical_compatible"
    assert first_entry["supersession_reason"] == "newer_run_recorded"
    assert "stale_reasons" not in first_entry
    assert first_entry["superseded_by"] == second_report["run_id"]
    assert first_path.read_bytes() == first_bytes


@pytest.mark.parametrize(
    ("section", "reason"),
    [
        ("dependencies", "dependency_environment_changed"),
        ("hardware", "hardware_changed"),
        ("artifacts", "artifacts_changed"),
        ("protocol", "protocol_changed"),
    ],
)
def test_history_compatibility_covers_complete_benchmark_provenance(section, reason) -> None:
    current = {
        "schema_version": "performance_baseline.v2",
        "corpus": {"sha256": "corpus"},
        "source": {"source_fingerprint": {"sha256": "source"}},
        "dependencies": {"definitions": {"poetry_lock": {"sha256": "lock"}}},
        "hardware": {"cuda": {"driver_version": "driver-a"}},
        "artifacts": {"detector_model": {"sha256": "model"}},
        "protocol": {"warm_repeats_per_page": 5},
    }
    previous = copy.deepcopy(current)
    previous[section]["mutation"] = True

    reasons = performance_module._history_stale_reasons(
        "performance_manifest.v2",
        previous,
        current,
    )

    assert reason in reasons


def test_history_compatibility_detects_source_fingerprint_change() -> None:
    current = {
        "schema_version": "performance_baseline.v2",
        "corpus": {"sha256": "corpus"},
        "source": {"source_fingerprint": {"sha256": "source-a"}},
        "dependencies": {},
        "hardware": {},
        "artifacts": {},
        "protocol": {},
    }
    previous = copy.deepcopy(current)
    previous["source"]["source_fingerprint"]["sha256"] = "source-b"

    reasons = performance_module._history_stale_reasons(
        "performance_manifest.v2",
        previous,
        current,
    )

    assert "source_changed" in reasons
