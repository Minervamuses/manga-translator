from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manga_translator.benchmark import detector_parity
from manga_translator.benchmark import cli as benchmark_cli
from manga_translator.benchmark.detector_parity import (
    DetectorParityBlocked,
    run_detector_parity,
)
from manga_translator.detector import DetectionResult, TextRegion

ROOT = Path(__file__).resolve().parents[1]


def _settings(path: Path) -> bytes:
    payload = {
        "schema_version": "detector_fp16_parity.v1",
        "required_pages": 5,
        "thresholds": {
            "box_count_ratio_min": 0.9,
            "box_mean_iou_min": 0.85,
            "matched_score_mae_max": 0.05,
            "mask_iou_min": 0.9,
            "small_text_recall_min": 0.9,
            "small_text_max_area_ratio": 0.0025,
        },
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    return encoded


def _matching_runner(pages, detection, postprocess, half):
    del detection, postprocess, half
    results = []
    for _path, image in pages:
        region = TextRegion(id="r1", x=2, y=2, w=10, h=10, confidence=0.95)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2:12, 2:12] = 255
        results.append(
            DetectionResult(regions_raw=[region], regions_post=[region], groups=[], mask=mask)
        )
    return results


def test_real_parity_runner_atomically_publishes_complete_evidence(tmp_path: Path) -> None:
    output = tmp_path / "parity.json"
    _settings(output)
    written, report = run_detector_parity(ROOT, output=output, runner=_matching_runner)

    assert written == output
    assert report["status"] == "passed"
    assert report["mock"] is False
    assert report["metrics"]["box_count_ratio"] == 1.0
    assert report["checks"] == {
        "box_count_ratio": True,
        "box_mean_iou": True,
        "matched_score_mae": True,
        "mask_iou": True,
        "small_text_recall": True,
    }
    assert len(report["pages"]) == 5
    assert json.loads(output.read_text(encoding="utf-8"))["run_id"] == report["run_id"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_blocked_parity_does_not_replace_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "parity.json"
    original = _settings(output)
    monkeypatch.setattr(detector_parity.torch.cuda, "is_available", lambda: False)

    with pytest.raises(DetectorParityBlocked, match="target_cuda_unavailable"):
        run_detector_parity(ROOT, output=output)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_partial_runner_failure_does_not_replace_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "parity.json"
    original = _settings(output)

    def incomplete(pages, detection, postprocess, half):
        return _matching_runner(pages, detection, postprocess, half)[:-1]

    with pytest.raises(DetectorParityBlocked, match="incomplete"):
        run_detector_parity(ROOT, output=output, runner=incomplete)

    assert output.read_bytes() == original


def test_require_real_cli_is_nonzero_when_parity_is_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        benchmark_cli,
        "run_detector_parity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DetectorParityBlocked(("target_cuda_unavailable",))
        ),
    )

    exit_code = benchmark_cli.main(["detector-parity", "--require-real"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_performance_require_real_requests_real_execution_and_is_nonzero_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def fake_run(root, profile, *, execute_real):
        observed.update(root=root, profile=profile, execute_real=execute_real)
        return tmp_path / "run.json", {
            "run_id": "blocked-run",
            "real_run": {"status": "blocked"},
        }

    monkeypatch.setattr(benchmark_cli, "run_performance_baseline", fake_run)

    exit_code = benchmark_cli.main(
        ["--root", str(tmp_path), "performance", "--require-real"]
    )

    assert exit_code == 1
    assert observed["execute_real"] is True
