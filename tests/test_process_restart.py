from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "fixtures" / "p1_restart_worker.py"
KILL_POINTS = (
    "detect-after",
    "ocr-after",
    "provider-response-after",
    "render-before",
)
EXPECTED_COUNTS = {
    "detector_load": 1,
    "detector_forward": 1,
    "ocr_load": 1,
    "ocr_forward": 1,
    "provider_request": 1,
}


def _worker(
    *,
    shared: Path,
    state: Path,
    kill_point: str = "none",
    mode: str = "run",
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment.pop("OPENROUTER_API_KEY", None)
    return subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--shared",
            str(shared),
            "--state",
            str(state),
            "--kill-point",
            kill_point,
            "--mode",
            mode,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_real_process_kills_resume_without_repeating_completed_work(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    uninterrupted_state = tmp_path / "uninterrupted"
    uninterrupted = _worker(shared=shared, state=uninterrupted_state)
    assert uninterrupted.returncode == 0, uninterrupted.stderr
    reference = json.loads(
        (uninterrupted_state / "result.json").read_text(encoding="utf-8")
    )
    assert reference["status"] == "succeeded"
    assert reference["counts"] == EXPECTED_COUNTS

    for kill_point in KILL_POINTS:
        state = tmp_path / kill_point
        killed = _worker(shared=shared, state=state, kill_point=kill_point)
        assert killed.returncode == 91, (kill_point, killed.stdout, killed.stderr)
        assert (state / f"killed-{kill_point}").read_text("utf-8") == kill_point

        time.sleep(0.35)
        resumed = _worker(shared=shared, state=state, kill_point=kill_point)
        assert resumed.returncode == 0, (kill_point, resumed.stdout, resumed.stderr)
        result = json.loads((state / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "succeeded"
        assert result["counts"] == EXPECTED_COUNTS
        assert result["document_sha256"] == reference["document_sha256"]

        replay = _worker(shared=shared, state=state, mode="replay")
        assert replay.returncode == 0, (kill_point, replay.stdout, replay.stderr)
        replay_result = json.loads(
            (state / "replay-result.json").read_text(encoding="utf-8")
        )
        assert replay_result["document_sha256"] == reference["document_sha256"]
        assert replay_result["network_access"] == 0
        assert replay_result["model_loads"] == 0
