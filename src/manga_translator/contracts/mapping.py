"""Immutable request-to-region mappings and exact response validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MAPPING_CHAIN_KEYS = (
    "region",
    "ocr_record",
    "request_item",
    "raw_response_item",
    "validated_translation",
    "layout_plan",
    "render_target",
)


def mapping_chain_template(
    *,
    region_key: str | None = None,
    ocr_record: str | None = None,
    request_item: str | None = None,
) -> dict[str, Any]:
    """Return a complete, null-filled seven-stage mapping chain."""
    return {
        "region": region_key,
        "ocr_record": ocr_record,
        "request_item": request_item,
        "raw_response_item": None,
        "validated_translation": None,
        "layout_plan": None,
        "render_target": None,
    }


def normalize_mapping_chain(chain: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the manifest contract stable even when a stage was not reached."""
    material = chain or {}
    return {key: material.get(key) for key in MAPPING_CHAIN_KEYS}


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MappingIssue:
    code: str
    details: Mapping[str, Any]


class MappingContractError(ValueError):
    def __init__(
        self,
        issues: Iterable[MappingIssue],
        *,
        raw_response_refs: Iterable[RawResponseRef] = (),
    ) -> None:
        self.issues = tuple(issues)
        self.raw_response_refs = tuple(dict.fromkeys(raw_response_refs))
        codes = ",".join(issue.code for issue in self.issues)
        super().__init__(f"translation mapping rejected: {codes}")


@dataclass(frozen=True)
class RequestItem:
    item_id: str
    region_key: str
    source_text: str
    source_sha256: str


@dataclass(frozen=True)
class RequestMap:
    request_id: str
    page_id: str
    items: tuple[RequestItem, ...]

    @property
    def by_item_id(self) -> Mapping[str, RequestItem]:
        return MappingProxyType({item.item_id: item for item in self.items})

    @property
    def by_region_key(self) -> Mapping[str, RequestItem]:
        return MappingProxyType({item.region_key: item for item in self.items})


@dataclass(frozen=True)
class RawResponseRef:
    sha256: str
    media_type: str
    size_bytes: int
    relative_path: str | None = None

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        media_type: str,
        relative_path: str | None = None,
    ) -> RawResponseRef:
        return cls(
            sha256=hashlib.sha256(payload).hexdigest(),
            media_type=media_type,
            size_bytes=len(payload),
            relative_path=relative_path,
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "sha256": self.sha256,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class ResponseItem:
    item_id: str
    source_sha256: str
    translation: str
    response_index: int
    raw_response_ref: RawResponseRef | None = None


@dataclass(frozen=True)
class ValidatedTranslationBatch:
    request: RequestMap
    responses: tuple[ResponseItem, ...]

    @property
    def by_region_key(self) -> Mapping[str, str]:
        response_by_id = {item.item_id: item.translation for item in self.responses}
        return MappingProxyType(
            {item.region_key: response_by_id[item.item_id] for item in self.request.items}
        )

    def chain_for(self, region_key: str) -> dict[str, Any]:
        request_item = self.request.by_region_key[region_key]
        response_item = next(
            response for response in self.responses if response.item_id == request_item.item_id
        )
        chain = mapping_chain_template(
            region_key=region_key,
            ocr_record=f"ocr:{region_key}",
            request_item=request_item.item_id,
        )
        chain["raw_response_item"] = {
            "item_id": response_item.item_id,
            "response_index": response_item.response_index,
            "artifact": (
                response_item.raw_response_ref.to_dict()
                if response_item.raw_response_ref is not None
                else None
            ),
        }
        chain["validated_translation"] = hashlib.sha256(
            response_item.translation.encode("utf-8")
        ).hexdigest()
        return chain


def build_request_map(
    page_id: str, units: Iterable[tuple[str, str]], *, request_id: str | None = None
) -> RequestMap:
    material = list(units)
    region_keys = [region_key for region_key, _source in material]
    if any(not key for key in region_keys) or len(set(region_keys)) != len(region_keys):
        raise MappingContractError(
            [MappingIssue("invalid_region_keys", {"region_keys": region_keys})]
        )
    if any(not isinstance(source, str) or not source.strip() for _key, source in material):
        raise MappingContractError([MappingIssue("empty_source", {})])
    if request_id is None:
        canonical = json.dumps(
            {
                "page_id": page_id,
                "units": [
                    {"region_key": key, "source_sha256": source_sha256(source)}
                    for key, source in material
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        request_id = "R" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    items = tuple(
        RequestItem(
            item_id=f"{request_id}:T{index:04d}",
            region_key=region_key,
            source_text=source,
            source_sha256=source_sha256(source),
        )
        for index, (region_key, source) in enumerate(material)
    )
    return RequestMap(request_id=request_id, page_id=page_id, items=items)


def request_map_from_ids(
    item_ids: list[str], source_hashes: list[str], *, request_id: str = "legacy"
) -> RequestMap:
    if len(item_ids) != len(source_hashes) or len(set(item_ids)) != len(item_ids):
        raise MappingContractError([MappingIssue("count_mismatch", {})])
    items = tuple(
        RequestItem(
            item_id=item_id,
            region_key=f"legacy:{index}",
            source_text="",
            source_sha256=source_hashes[index],
        )
        for index, item_id in enumerate(item_ids)
    )
    return RequestMap(request_id=request_id, page_id="legacy", items=items)


def validate_response_items(
    request: RequestMap,
    raw_items: Any,
    *,
    raw_response_ref: RawResponseRef | None = None,
) -> ValidatedTranslationBatch:
    issues: list[MappingIssue] = []
    if not isinstance(raw_items, list):
        raise MappingContractError(
            [MappingIssue("invalid_response_type", {"actual": type(raw_items).__name__})]
        )

    expected = request.by_item_id
    seen: set[str] = set()
    responses: list[ResponseItem] = []
    unknown: list[str] = []
    duplicates: list[str] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            issues.append(MappingIssue("invalid_item_type", {"index": index}))
            continue
        item_id = raw.get("id")
        if not isinstance(item_id, str):
            issues.append(MappingIssue("invalid_id_type", {"index": index}))
            continue
        if item_id in seen:
            duplicates.append(item_id)
            continue
        seen.add(item_id)
        if item_id not in expected:
            unknown.append(item_id)
            continue
        translation = raw.get("translation", raw.get("text"))
        if not isinstance(translation, str):
            issues.append(MappingIssue("invalid_translation_type", {"id": item_id}))
            continue
        if not translation.strip():
            issues.append(MappingIssue("empty_translation", {"id": item_id}))
            continue
        response_source_hash = raw.get("source_sha256")
        if not isinstance(response_source_hash, str):
            issues.append(MappingIssue("missing_source_hash", {"id": item_id}))
            continue
        if response_source_hash != expected[item_id].source_sha256:
            issues.append(MappingIssue("source_binding_mismatch", {"id": item_id}))
            continue
        responses.append(
            ResponseItem(
                item_id=item_id,
                source_sha256=response_source_hash,
                translation=translation.strip(),
                response_index=index,
                raw_response_ref=raw_response_ref,
            )
        )

    if duplicates:
        issues.append(MappingIssue("duplicate_id", {"ids": sorted(set(duplicates))}))
    if unknown:
        issues.append(MappingIssue("unknown_id", {"ids": sorted(set(unknown))}))
        issues.append(MappingIssue("extra_id", {"ids": sorted(set(unknown))}))
    missing = sorted(set(expected) - seen)
    if missing:
        issues.append(MappingIssue("missing_id", {"ids": missing}))
    if len(raw_items) != len(request.items):
        issues.append(
            MappingIssue(
                "count_mismatch",
                {"expected": len(request.items), "actual": len(raw_items)},
            )
        )
    if issues or len(responses) != len(request.items):
        if not issues:
            issues.append(MappingIssue("invalid_response_items", {}))
        raise MappingContractError(
            issues,
            raw_response_refs=([raw_response_ref] if raw_response_ref is not None else []),
        )

    response_by_id = {item.item_id: item for item in responses}
    ordered = tuple(response_by_id[item.item_id] for item in request.items)
    return ValidatedTranslationBatch(request=request, responses=ordered)


def bind_validated_responses(
    request: RequestMap,
    responses: Iterable[ResponseItem],
) -> ValidatedTranslationBatch:
    material = tuple(responses)
    issues: list[MappingIssue] = []
    expected = request.by_item_id
    response_by_id: dict[str, ResponseItem] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for response in material:
        if response.item_id in response_by_id:
            duplicates.append(response.item_id)
            continue
        response_by_id[response.item_id] = response
        request_item = expected.get(response.item_id)
        if request_item is None:
            unknown.append(response.item_id)
            continue
        if response.source_sha256 != request_item.source_sha256:
            issues.append(MappingIssue("source_binding_mismatch", {"id": response.item_id}))
        if not response.translation.strip():
            issues.append(MappingIssue("empty_translation", {"id": response.item_id}))

    if duplicates:
        issues.append(MappingIssue("duplicate_id", {"ids": sorted(set(duplicates))}))
    if unknown:
        issues.append(MappingIssue("unknown_id", {"ids": sorted(set(unknown))}))
        issues.append(MappingIssue("extra_id", {"ids": sorted(set(unknown))}))
    missing = sorted(set(expected) - set(response_by_id))
    if missing:
        issues.append(MappingIssue("missing_id", {"ids": missing}))
    if len(material) != len(request.items):
        issues.append(
            MappingIssue(
                "count_mismatch",
                {"expected": len(request.items), "actual": len(material)},
            )
        )
    if issues:
        raise MappingContractError(
            issues,
            raw_response_refs=(
                response.raw_response_ref
                for response in material
                if response.raw_response_ref is not None
            ),
        )
    ordered = tuple(response_by_id[item.item_id] for item in request.items)
    return ValidatedTranslationBatch(request=request, responses=ordered)


def bind_validated_values(request: RequestMap, values: list[str]) -> ValidatedTranslationBatch:
    raw_items = [
        {
            "id": item.item_id,
            "source_sha256": item.source_sha256,
            "translation": values[index] if index < len(values) else None,
        }
        for index, item in enumerate(request.items)
    ]
    if len(values) > len(request.items):
        raw_items.extend(
            {
                "id": f"{request.request_id}:EXTRA{index:04d}",
                "source_sha256": "",
                "translation": value,
            }
            for index, value in enumerate(values[len(request.items) :])
        )
    return validate_response_items(request, raw_items)
