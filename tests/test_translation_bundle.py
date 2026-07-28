from __future__ import annotations

import base64
import json

import pytest

from manga_translator.contracts.mapping import (
    MappingContractError,
    MappingIssue,
    RawResponseRef,
    ResponseItem,
    bind_validated_responses,
    build_request_map,
)
from manga_translator.pipeline import (
    TranslationBundleReplayError,
    _deserialize_translation_bundle,
    _serialize_translation_bundle,
    _serialize_translation_failure_bundle,
)


def _bundle(tmp_path):
    request = build_request_map("a" * 64, [("region:1", "猫")])
    raw = b'{"choices":[{"message":{"content":"translated"}}]}'
    reference = RawResponseRef.from_bytes(
        raw,
        media_type="application/json",
        relative_path="artifacts/translation-responses/raw.json",
    )
    path = tmp_path / reference.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    response = ResponseItem(
        item_id=request.items[0].item_id,
        source_sha256=request.items[0].source_sha256,
        translation="貓",
        response_index=0,
        raw_response_ref=reference,
    )
    return bind_validated_responses(request, [response]), raw, path


def test_translation_bundle_replays_exact_raw_provider_bytes_without_source_file(
    tmp_path,
) -> None:
    batch, raw, path = _bundle(tmp_path)
    serialized = _serialize_translation_bundle(batch, artifact_root=tmp_path)
    path.unlink()

    restored, raw_artifacts = _deserialize_translation_bundle(serialized)

    assert restored == batch
    reference = restored.responses[0].raw_response_ref
    assert reference is not None
    assert raw_artifacts[reference.sha256] == (reference, raw)


def test_translation_bundle_rejects_tampered_embedded_provider_response(tmp_path) -> None:
    batch, _raw, _path = _bundle(tmp_path)
    payload = json.loads(_serialize_translation_bundle(batch, artifact_root=tmp_path))
    payload["raw_responses"][0]["payload_base64"] = base64.b64encode(
        b"tampered"
    ).decode("ascii")

    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        _deserialize_translation_bundle(json.dumps(payload).encode("utf-8"))


def test_translation_bundle_rejects_provider_path_escape(tmp_path) -> None:
    batch, _raw, _path = _bundle(tmp_path)
    response = batch.responses[0]
    assert response.raw_response_ref is not None
    escaped = RawResponseRef(
        sha256=response.raw_response_ref.sha256,
        media_type=response.raw_response_ref.media_type,
        size_bytes=response.raw_response_ref.size_bytes,
        relative_path="../outside.json",
    )
    unsafe = bind_validated_responses(
        batch.request,
        [
            ResponseItem(
                item_id=response.item_id,
                source_sha256=response.source_sha256,
                translation=response.translation,
                response_index=response.response_index,
                raw_response_ref=escaped,
            )
        ],
    )

    with pytest.raises(ValueError, match="remain below"):
        _serialize_translation_bundle(unsafe, artifact_root=tmp_path)


def test_translation_bundle_deduplicates_shared_raw_provider_bytes(tmp_path) -> None:
    request = build_request_map(
        "b" * 64,
        [("region:1", "猫"), ("region:2", "犬")],
    )
    raw = b'{"shared":"provider response"}'
    reference = RawResponseRef.from_bytes(
        raw,
        media_type="application/json",
        relative_path="raw/shared.json",
    )
    path = tmp_path / "raw" / "shared.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    batch = bind_validated_responses(
        request,
        [
            ResponseItem(
                item_id=item.item_id,
                source_sha256=item.source_sha256,
                translation=f"translation-{index}",
                response_index=index,
                raw_response_ref=reference,
            )
            for index, item in enumerate(request.items)
        ],
    )

    payload = json.loads(_serialize_translation_bundle(batch, artifact_root=tmp_path))
    restored, raw_artifacts = _deserialize_translation_bundle(
        json.dumps(payload).encode("utf-8")
    )

    assert len(payload["raw_responses"]) == 1
    assert restored == batch
    assert raw_artifacts == {reference.sha256: (reference, raw)}


def test_rejected_provider_response_bundle_replays_without_source_file(tmp_path) -> None:
    batch, raw, path = _bundle(tmp_path)
    reference = batch.responses[0].raw_response_ref
    assert reference is not None
    error = MappingContractError(
        [MappingIssue("missing_id", {"ids": [batch.request.items[0].item_id]})],
        raw_response_refs=[reference],
    )
    serialized = _serialize_translation_failure_bundle(
        batch.request,
        error,
        artifact_root=tmp_path,
    )
    path.unlink()

    with pytest.raises(TranslationBundleReplayError) as captured:
        _deserialize_translation_bundle(serialized)

    assert captured.value.mapping_issues == error.issues
    assert captured.value.raw_artifacts == {reference.sha256: (reference, raw)}
