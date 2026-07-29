from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "benchmarks" / "gates" / "validate_g0.py"
SPEC = importlib.util.spec_from_file_location("validate_g0", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _payloads():
    gate = json.loads(
        (ROOT / "benchmarks/gates/g0_correctness_baseline.json").read_text("utf-8")
    )
    parity = json.loads((ROOT / gate["artifacts"]["detector_parity"]["path"]).read_text("utf-8"))
    performance = json.loads(
        (ROOT / gate["artifacts"]["performance"]["path"]).read_text("utf-8")
    )
    return gate, parity, performance


def test_checked_in_g0_gate_locks_a_valid_historical_baseline() -> None:
    assert VALIDATOR.validate_g0(ROOT) == {"gate": "G0", "status": "passed", "errors": []}


def test_g0_rejects_missing_reviewer() -> None:
    gate, parity, performance = _payloads()
    gate["reviewers"] = []

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, performance)

    assert "ground truth: reviewer declaration missing" in errors


def test_g0_rejects_missing_real_metrics() -> None:
    gate, parity, performance = _payloads()
    performance["real_run"]["summary"] = None

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, performance)

    assert "performance: required real metrics missing" in errors


def test_g0_rejects_threshold_failure() -> None:
    gate, parity, performance = _payloads()
    parity["metrics"]["mask_iou"] = 0.0

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, performance)

    assert "detector parity: one or more thresholds failed" in errors


def test_g0_rejects_mock_as_real() -> None:
    gate, parity, performance = _payloads()
    parity["mock"] = True

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, performance)

    assert "detector parity: mock evidence cannot satisfy G0" in errors


def test_g0_rejects_mock_sample_inside_real_measurements() -> None:
    gate, parity, performance = _payloads()
    performance["real_run"]["measurements"][0]["profiler"]["environment_kind"] = "mock"

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, performance)

    assert "performance: mock or incomplete sample found in real measurements" in errors


def test_g0_rejects_stale_fingerprint() -> None:
    gate, parity, performance = _payloads()
    stale = copy.deepcopy(performance)
    stale["source"]["source_fingerprint"]["sha256"] = "0" * 64

    errors = VALIDATOR.validate_payloads(ROOT, gate, parity, stale)

    assert "performance: stale source fingerprint" in errors
