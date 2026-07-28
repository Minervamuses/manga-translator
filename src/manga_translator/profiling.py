"""Low-overhead, opt-in profiling for pipeline and benchmark runs."""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import Any

import torch


@dataclass(frozen=True)
class SpanRecord:
    sequence: int
    stage: str
    page_id: str | None
    wall_ms: float
    gpu_ms: float | None
    rss_before_bytes: int | None
    rss_after_bytes: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@cache
def _windows_rss_reader() -> Callable[[], int | None]:
    import ctypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    handle = get_current_process()

    def read() -> int | None:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = get_process_memory_info(handle, ctypes.byref(counters), counters.cb)
        return int(counters.working_set_size) if ok else None

    return read


def current_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            return _windows_rss_reader()()
        except (AttributeError, OSError):
            return None
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if platform.system() == "Darwin" else rss * 1024
    except (ImportError, OSError):
        return None


class RunProfiler:
    def __init__(
        self,
        run_id: str,
        *,
        enabled: bool = True,
        environment_kind: str = "real",
    ) -> None:
        self.run_id = run_id
        self.enabled = enabled
        self.environment_kind = environment_kind
        self.started_at = datetime.now(UTC).isoformat()
        self.finished_at: str | None = None
        self.spans: list[SpanRecord] = []
        self.pages: dict[str, dict[str, Any]] = {}
        self.api_usage: list[dict[str, Any]] = []
        self._sequence = 0
        self._rss_peak = current_rss_bytes()
        if enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def span(
        self,
        stage: str,
        *,
        gpu: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        page_id = _ACTIVE_PAGE.get()
        rss_before = current_rss_bytes()
        started_ns = time.perf_counter_ns()
        start_event = None
        end_event = None
        if gpu and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            gpu_ms: float | None = None
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                gpu_ms = float(start_event.elapsed_time(end_event))
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            rss_after = current_rss_bytes()
            observed = [value for value in (self._rss_peak, rss_before, rss_after) if value]
            self._rss_peak = max(observed) if observed else None
            self._sequence += 1
            self.spans.append(
                SpanRecord(
                    sequence=self._sequence,
                    stage=stage,
                    page_id=page_id,
                    wall_ms=elapsed_ms,
                    gpu_ms=gpu_ms,
                    rss_before_bytes=rss_before,
                    rss_after_bytes=rss_after,
                    metadata=dict(metadata or {}),
                )
            )

    @contextmanager
    def page(self, page_id: str, source_path: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if _ACTIVE_PAGE.get() == page_id:
            self.pages.setdefault(page_id, {"source_path": source_path})
            yield
            return
        token = _ACTIVE_PAGE.set(page_id)
        self.pages.setdefault(page_id, {"source_path": source_path})
        try:
            with self.span("page_wall"):
                yield
        finally:
            _ACTIVE_PAGE.reset(token)

    def set_page_metrics(self, page_id: str, **metrics: Any) -> None:
        if self.enabled:
            self.pages.setdefault(page_id, {}).update(metrics)

    def record_api_usage(
        self,
        *,
        model: str,
        status_code: int,
        latency_ms: float,
        usage: dict[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        self.api_usage.append(
            {
                "page_id": _ACTIVE_PAGE.get(),
                "model": model,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "usage": dict(usage or {}),
            }
        )

    def finish(self) -> dict[str, Any]:
        self.finished_at = datetime.now(UTC).isoformat()
        cuda_memory: dict[str, int] | None = None
        if torch.cuda.is_available():
            cuda_memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        return {
            "schema_version": "pipeline_profile.v1",
            "run_id": self.run_id,
            "enabled": self.enabled,
            "environment_kind": self.environment_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pages": self.pages,
            "spans": [asdict(span) for span in self.spans],
            "api_usage": self.api_usage,
            "resources": {
                "cpu_rss_peak_bytes": self._rss_peak,
                "cuda": cuda_memory,
            },
        }


_ACTIVE_PROFILER: ContextVar[RunProfiler | None] = ContextVar(
    "manga_translator_active_profiler", default=None
)
_ACTIVE_PAGE: ContextVar[str | None] = ContextVar(
    "manga_translator_active_profile_page", default=None
)


@contextmanager
def activate_profiler(profiler: RunProfiler) -> Iterator[RunProfiler]:
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def profile_span(
    stage: str,
    *,
    gpu: bool = False,
    **metadata: Any,
) -> Iterator[None]:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None or not profiler.enabled:
        yield
        return
    with profiler.span(stage, gpu=gpu, metadata=metadata):
        yield


@contextmanager
def profile_page(page_id: str, source_path: str) -> Iterator[None]:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None or not profiler.enabled:
        yield
        return
    with profiler.page(page_id, source_path):
        yield


def set_page_profile_metrics(page_id: str, **metrics: Any) -> None:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.set_page_metrics(page_id, **metrics)


def record_api_profile(
    *,
    model: str,
    status_code: int,
    latency_ms: float,
    usage: dict[str, Any] | None,
) -> None:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.record_api_usage(
            model=model,
            status_code=status_code,
            latency_ms=latency_ms,
            usage=usage,
        )


def measure_profiler_overhead(iterations: int = 1_000) -> dict[str, float]:
    iterations = max(1, int(iterations))
    started = time.perf_counter_ns()
    for _ in range(iterations):
        pass
    loop_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    for _ in range(iterations):
        with profile_span("disabled_probe"):
            pass
    disabled_ns = time.perf_counter_ns() - started

    profiler = RunProfiler("overhead-probe", environment_kind="mock")
    started = time.perf_counter_ns()
    with activate_profiler(profiler):
        for _ in range(iterations):
            with profile_span("enabled_probe"):
                pass
    enabled_ns = time.perf_counter_ns() - started
    return {
        "iterations": float(iterations),
        "disabled_ns_per_span": max(0.0, (disabled_ns - loop_ns) / iterations),
        "enabled_ns_per_span": max(0.0, (enabled_ns - loop_ns) / iterations),
    }
