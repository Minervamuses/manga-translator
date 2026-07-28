from __future__ import annotations

import json

import pytest

from manga_translator.contracts.mapping import (
    MappingContractError,
    build_request_map,
    validate_response_items,
)
from manga_translator.contracts.translation import parse_translation_response


def _request(page_id: str = "page-a"):
    return build_request_map(page_id, [("r-a", "猫だ"), ("r-b", "犬だ")])


def _valid_items(request):
    return [
        {
            "id": item.item_id,
            "source_sha256": item.source_sha256,
            "translation": translation,
        }
        for item, translation in zip(request.items, ("是貓", "是狗"))
    ]


def _codes(error: MappingContractError) -> set[str]:
    return {issue.code for issue in error.issues}


def test_response_array_reorder_preserves_region_mapping() -> None:
    request = _request()
    batch = validate_response_items(request, list(reversed(_valid_items(request))))
    assert batch.by_region_key == {"r-a": "是貓", "r-b": "是狗"}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda items, _other: [items[0], items[0]], "duplicate_id"),
        (lambda items, _other: items[:1], "missing_id"),
        (
            lambda items, _other: items
            + [{"id": "extra", "source_sha256": "0" * 64, "translation": "額外"}],
            "extra_id",
        ),
        (
            lambda items, _other: [items[0], {**items[1], "id": "unknown"}],
            "unknown_id",
        ),
        (
            lambda items, _other: [{**items[0], "id": items[0]["id"].lower()}, items[1]],
            "unknown_id",
        ),
        (
            lambda items, _other: [items[0], {**items[0], "translation": "後值"}],
            "duplicate_id",
        ),
        (
            lambda items, other: [items[0], {**items[1], "id": other.items[1].item_id}],
            "unknown_id",
        ),
        (
            lambda items, _other: [
                {**items[0], "source_sha256": items[1]["source_sha256"], "translation": "是狗"},
                {**items[1], "source_sha256": items[0]["source_sha256"], "translation": "是貓"},
            ],
            "source_binding_mismatch",
        ),
    ],
)
def test_destructive_mapping_mutations_reject_whole_batch(mutation, expected_code) -> None:
    request = _request()
    other_page = _request("page-b")
    with pytest.raises(MappingContractError) as captured:
        validate_response_items(request, mutation(_valid_items(request), other_page))
    assert expected_code in _codes(captured.value)


def test_malformed_json_and_partial_response_are_rejected() -> None:
    request = _request()
    with pytest.raises(MappingContractError) as malformed:
        parse_translation_response("not json", request)
    assert "malformed_json" in _codes(malformed.value)

    partial = json.dumps({"translations": _valid_items(request)[:1]}, ensure_ascii=False)
    with pytest.raises(MappingContractError) as missing:
        parse_translation_response(partial, request)
    assert {"missing_id", "count_mismatch"} <= _codes(missing.value)


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [("", "empty_translation"), (42, "invalid_translation_type")],
)
def test_translation_value_must_be_a_nonempty_string(value, expected_code) -> None:
    request = _request()
    items = _valid_items(request)
    items[0]["translation"] = value
    with pytest.raises(MappingContractError) as captured:
        validate_response_items(request, items)
    assert expected_code in _codes(captured.value)


def test_request_maps_are_immutable_and_page_local() -> None:
    first = _request("page-a")
    second = _request("page-b")
    assert first.items[0].item_id != second.items[0].item_id
    with pytest.raises(TypeError):
        first.by_item_id[first.items[0].item_id] = first.items[1]


def test_mapping_chain_covers_request_through_render_placeholders() -> None:
    request = _request()
    batch = validate_response_items(request, _valid_items(request))
    chain = batch.chain_for("r-a")
    assert chain["region"] == "r-a"
    assert chain["request_item"] == request.items[0].item_id
    assert chain["raw_response_item"] == request.items[0].item_id
    assert chain["validated_translation"]
    assert chain["layout_plan"] is None
    assert chain["render_target"] is None
