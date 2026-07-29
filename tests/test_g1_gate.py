from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_g1_records_all_real_restart_and_offline_replay_evidence() -> None:
    gate = json.loads(
        (ROOT / "benchmarks/gates/g1_page_document.json").read_text("utf-8")
    )

    assert gate["schema_version"] == "g1_page_document.v2"
    assert gate["status"] == "passed"
    assert gate["deferred_validation"] == []
    assert gate["blockers"] == []
    assert not any(
        "pending" in str(value) for value in gate["verification"].values()
    )
    restart = gate["process_restart"]
    assert restart["execution_kind"] == "real_os_subprocess_kill_and_restart"
    assert restart["kill_points"] == [
        "detect-after",
        "ocr-after",
        "provider-response-after",
        "render-before",
    ]
    assert restart["expected_total_calls_after_resume"] == {
        "detector_load": 1,
        "detector_forward": 1,
        "ocr_load": 1,
        "ocr_forward": 1,
        "provider_request": 1,
    }
    assert restart["canonical_document_hash"].startswith("identical")
    assert restart["offline_replay"] == {
        "network_access": 0,
        "model_loads": 0,
        "document_hash": "identical_to_durable_page_document",
        "encoded_image": "replayed_from_content_addressed_artifact",
    }
