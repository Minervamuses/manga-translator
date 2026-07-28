from __future__ import annotations

import hashlib
import json
from pathlib import Path

from manga_translator.benchmark import performance as performance_module
from manga_translator.benchmark.performance import REQUIRED_STAGES, run_performance_baseline
from manga_translator.profiling import (
    RunProfiler,
    activate_profiler,
    measure_profiler_overhead,
    profile_page,
    profile_span,
    record_api_profile,
    set_page_profile_metrics,
)


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
        payload = {
            "page_id": hashlib.sha256(source.read_bytes()).hexdigest(),
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
        "openrouter:\n  api_key: YOUR_OPENROUTER_API_KEY\n  model: test/model\n",
        encoding="utf-8",
    )
    return tmp_path


def test_performance_baseline_separates_mock_from_blocked_real_run(
    tmp_path,
    monkeypatch,
) -> None:
    root = _benchmark_root(tmp_path)
    monkeypatch.setattr(
        performance_module,
        "_power_profile",
        lambda: {"status": "unavailable", "value": "test"},
    )

    run_path, report = run_performance_baseline(root)

    assert run_path.is_file()
    assert report["corpus"]["sha256"]
    assert len(report["corpus"]["pages"]) == 5
    assert report["mock_run"]["environment_kind"] == "mock"
    assert report["mock_run"]["status"] == "complete"
    assert report["mock_run"]["warmup_runs"] == 1
    assert report["mock_run"]["repeats"] == 5
    assert report["mock_run"]["summary"]["p95_wall_ms"] >= 0
    assert report["mock_run"]["summary"]["worst_page"]["page_id"]
    assert report["real_run"]["environment_kind"] == "real"
    assert report["real_run"]["status"] == "blocked"
    assert report["real_run"]["measurements"] == []
    stages = {span["stage"] for span in report["mock_run"]["profiler"]["spans"]}
    assert set(REQUIRED_STAGES) <= stages

    manifest_path = root / "benchmarks" / "performance" / "v032_baseline" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["runs"][0]
    assert entry["run_id"] == report["run_id"]
    assert entry["sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest()
