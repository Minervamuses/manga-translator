from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from manga_translator.detector import DetectionResult, DetectorIssue, TextGroup, TextRegion
from manga_translator.domain.issues import StageName
from manga_translator.domain.models import ArtifactRef
from manga_translator.stages.state import (
    MASK_MEDIA_TYPE,
    STATE_MEDIA_TYPE,
    decode_pipeline_state,
    encode_pipeline_state,
)


def _detection() -> DetectionResult:
    local = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    region = TextRegion(
        id="r1",
        x=2,
        y=3,
        w=2,
        h=2,
        vertical=True,
        confidence=0.75,
        source="ctd",
        raw_index=4,
        detection_input_size=1024,
        font_size_hint=18.5,
        mask_bbox=(2, 3, 2, 2),
        local_mask=local,
        group_id="g1",
    )
    group = TextGroup(
        id="g1",
        region_ids=["r1"],
        bbox=(2, 3, 2, 2),
        vertical=True,
        ocr_text="猫",
        ocr_text_norm="猫",
        ocr_confidence=0.9,
        ocr_source="ensemble",
        ocr_candidates=[{"text": "猫", "score": 0.9}],
        status="ready",
        mapping_region_key="group:key",
        mapping_chain={"region": "group:key"},
        mask=local.copy(),
    )
    aggregate = np.zeros((8, 8), dtype=np.uint8)
    aggregate[3:5, 2:4] = local
    return DetectionResult(
        regions_raw=[region],
        regions_post=[region],
        groups=[group],
        mask=aggregate,
        raw_mask=aggregate.copy(),
        issues=[DetectorIssue("warning", "detector warning", {"count": 1})],
    )


def _materialize(outputs):
    content: dict[str, bytes] = {}
    refs: list[ArtifactRef] = []
    for payload in outputs.artifacts:
        sha256 = hashlib.sha256(payload.data).hexdigest()
        content[sha256] = payload.data
        refs.append(
            ArtifactRef(
                sha256=sha256,
                media_type=payload.media_type,
                size_bytes=len(payload.data),
            )
        )
    return tuple(refs), content


def _extras() -> dict[str, object]:
    return {
        "source_artifact": {
            "sha256": "a" * 64,
            "media_type": "image/png",
            "size_bytes": 10,
        }
    }


def test_pipeline_state_round_trip_is_deterministic_and_artifact_backed() -> None:
    extras = {**_extras(), "order": ["g1"]}
    first = encode_pipeline_state(
        _detection(), producer_stage=StageName.DETECT, extras=extras
    )
    second = encode_pipeline_state(
        _detection(), producer_stage=StageName.DETECT, extras=extras
    )
    assert first == second
    refs, content = _materialize(first)

    state_refs = [ref for ref in refs if ref.media_type == STATE_MEDIA_TYPE]
    mask_refs = [ref for ref in refs if ref.media_type == MASK_MEDIA_TYPE]
    assert len(state_refs) == 1
    assert len(mask_refs) == 2  # local/group masks deduplicate; aggregate mask deduplicates too
    metadata = json.loads(content[state_refs[0].sha256])
    assert metadata["producer_stage"] == "detect"
    assert len(metadata["detection"]["groups"][0]["stable_group_key"]) == 64
    assert metadata["detection"]["groups"][0]["mask"] in {ref.sha256 for ref in mask_refs}
    assert "data" not in metadata["detection"]["groups"][0]

    restored = decode_pipeline_state(
        refs,
        expected_stage=StageName.DETECT,
        read_bytes=content.__getitem__,
    )
    assert restored.extras == extras
    assert restored.detection.groups[0].ocr_text == "猫"
    assert np.array_equal(restored.detection.groups[0].mask, np.array([[0, 255], [255, 0]]))
    assert np.array_equal(restored.detection.mask, _detection().mask)


def test_pipeline_state_rejects_mask_reference_not_declared_by_stage_output() -> None:
    outputs = encode_pipeline_state(
        _detection(), producer_stage=StageName.DETECT, extras=_extras()
    )
    refs, content = _materialize(outputs)
    state_ref = next(ref for ref in refs if ref.media_type == STATE_MEDIA_TYPE)
    without_masks = (state_ref,)

    with pytest.raises(ValueError, match="undeclared mask artifact"):
        decode_pipeline_state(
            without_masks,
            expected_stage=StageName.DETECT,
            read_bytes=content.__getitem__,
        )


def test_pipeline_state_rejects_wrong_producer_stage() -> None:
    outputs = encode_pipeline_state(
        _detection(), producer_stage=StageName.DETECT, extras=_extras()
    )
    refs, content = _materialize(outputs)

    with pytest.raises(ValueError, match="producer mismatch"):
        decode_pipeline_state(
            refs,
            expected_stage=StageName.OCR,
            read_bytes=content.__getitem__,
        )


def test_pipeline_state_rejects_tampered_stable_group_key() -> None:
    outputs = encode_pipeline_state(
        _detection(), producer_stage=StageName.DETECT, extras=_extras()
    )
    refs, content = _materialize(outputs)
    state_ref = next(ref for ref in refs if ref.media_type == STATE_MEDIA_TYPE)
    payload = json.loads(content[state_ref.sha256])
    payload["detection"]["groups"][0]["stable_group_key"] = "0" * 64
    tampered = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    tampered_sha256 = hashlib.sha256(tampered).hexdigest()
    content[tampered_sha256] = tampered
    tampered_refs = tuple(
        ArtifactRef(
            sha256=tampered_sha256,
            media_type=reference.media_type,
            size_bytes=len(tampered),
        )
        if reference == state_ref
        else reference
        for reference in refs
    )

    with pytest.raises(ValueError, match="stable identity key mismatch"):
        decode_pipeline_state(
            tampered_refs,
            expected_stage=StageName.DETECT,
            read_bytes=content.__getitem__,
        )


def test_pipeline_state_rejects_missing_stage_specific_extras() -> None:
    with pytest.raises(ValueError, match="missing required extras"):
        encode_pipeline_state(
            _detection(),
            producer_stage=StageName.ORDER,
            extras=_extras(),
        )
