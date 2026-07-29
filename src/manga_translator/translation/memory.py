"""Approval-aware translation memory with context and entity revision binding."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from ..storage.job_store import JobStore

MemoryStatus = Literal["suggestion", "approved"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_nfc(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("translation memory source must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip():
        raise ValueError("translation memory source must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class MemoryKey:
    value: str
    source_nfc: str
    context_hash: str
    order_hash: str
    entity_revision_hash: str


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    key: MemoryKey
    target_zh_tw: str
    status: MemoryStatus
    reviewer_id: str | None
    provenance: dict[str, Any]

    @property
    def reusable(self) -> bool:
        return self.status == "approved" and bool((self.reviewer_id or "").strip())


def build_memory_key(
    source: str,
    *,
    context: Any,
    order: Any,
    entity_revision_hash: str,
) -> MemoryKey:
    source_nfc = _source_nfc(source)
    entity_revision_hash = _require_sha256(entity_revision_hash, "entity_revision_hash")
    context_hash = _canonical_hash(context)
    order_hash = _canonical_hash(order)
    value = _canonical_hash(
        {
            "source_nfc": source_nfc,
            "context_hash": context_hash,
            "order_hash": order_hash,
            "entity_revision_hash": entity_revision_hash,
        }
    )
    return MemoryKey(value, source_nfc, context_hash, order_hash, entity_revision_hash)


def _validate_memory_key(key: MemoryKey) -> None:
    if not isinstance(key, MemoryKey):
        raise TypeError("key must be a MemoryKey")
    source_nfc = _source_nfc(key.source_nfc)
    if source_nfc != key.source_nfc:
        raise ValueError("memory key source_nfc is not NFC-normalized")
    context_hash = _require_sha256(key.context_hash, "context_hash")
    order_hash = _require_sha256(key.order_hash, "order_hash")
    entity_hash = _require_sha256(key.entity_revision_hash, "entity_revision_hash")
    expected = _canonical_hash(
        {
            "source_nfc": source_nfc,
            "context_hash": context_hash,
            "order_hash": order_hash,
            "entity_revision_hash": entity_hash,
        }
    )
    if _require_sha256(key.value, "memory key") != expected:
        raise ValueError("memory key does not match its components")


class TranslationMemory:
    def __init__(self, store: JobStore, *, job_id: str, chapter_id: str) -> None:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id is required")
        if not isinstance(chapter_id, str) or not chapter_id.strip():
            raise ValueError("chapter_id is required")
        self.store = store
        self.job_id = job_id
        self.chapter_id = chapter_id

    def put(
        self,
        key: MemoryKey,
        target_zh_tw: str,
        *,
        status: MemoryStatus = "suggestion",
        reviewer_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> MemoryMatch:
        _validate_memory_key(key)
        if status not in {"suggestion", "approved"}:
            raise ValueError(f"unsupported translation memory status: {status}")
        if not isinstance(target_zh_tw, str):
            raise TypeError("translation memory target must be a string")
        target = unicodedata.normalize("NFC", target_zh_tw).strip()
        if not target:
            raise ValueError("translation memory target must not be empty")
        reviewer = reviewer_id.strip() if isinstance(reviewer_id, str) else None
        if status == "approved" and not reviewer:
            raise ValueError("approved translation memory requires reviewer_id")
        if provenance is not None and not isinstance(provenance, dict):
            raise TypeError("translation memory provenance must be an object")
        payload = json.dumps(
            {} if provenance is None else provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO translation_memory(
                    job_id, chapter_id, memory_key, source_nfc, context_hash, order_hash,
                    entity_revision_hash, target_zh_tw, status, reviewer_id, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, chapter_id, memory_key) DO UPDATE SET
                    target_zh_tw=CASE
                        WHEN translation_memory.status='approved' AND excluded.status='suggestion'
                        THEN translation_memory.target_zh_tw ELSE excluded.target_zh_tw END,
                    status=CASE
                        WHEN translation_memory.status='approved' AND excluded.status='suggestion'
                        THEN translation_memory.status ELSE excluded.status END,
                    reviewer_id=CASE
                        WHEN translation_memory.status='approved' AND excluded.status='suggestion'
                        THEN translation_memory.reviewer_id ELSE excluded.reviewer_id END,
                    provenance_json=CASE
                        WHEN translation_memory.status='approved' AND excluded.status='suggestion'
                        THEN translation_memory.provenance_json ELSE excluded.provenance_json END,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    self.job_id,
                    self.chapter_id,
                    key.value,
                    key.source_nfc,
                    key.context_hash,
                    key.order_hash,
                    key.entity_revision_hash,
                    target,
                    status,
                    reviewer,
                    payload,
                ),
            )
        match = self.lookup_key(key)
        assert match is not None
        return match

    def lookup_key(self, key: MemoryKey) -> MemoryMatch | None:
        _validate_memory_key(key)
        row = self.store.connection.execute(
            """
            SELECT target_zh_tw, status, reviewer_id, provenance_json
            FROM translation_memory
            WHERE job_id=? AND chapter_id=? AND memory_key=?
            """,
            (self.job_id, self.chapter_id, key.value),
        ).fetchone()
        if row is None:
            return None
        return MemoryMatch(
            key=key,
            target_zh_tw=str(row[0]),
            status=str(row[1]),  # type: ignore[arg-type]
            reviewer_id=str(row[2]) if row[2] is not None else None,
            provenance=json.loads(str(row[3])),
        )

    def lookup(
        self,
        source: str,
        *,
        context: Any,
        order: Any,
        entity_revision_hash: str,
    ) -> MemoryMatch | None:
        return self.lookup_key(
            build_memory_key(
                source,
                context=context,
                order=order,
                entity_revision_hash=entity_revision_hash,
            )
        )

    def suggestion_for_source(self, source: str) -> tuple[MemoryMatch, ...]:
        source_nfc = _source_nfc(source)
        rows = self.store.connection.execute(
            """
            SELECT memory_key, context_hash, order_hash, entity_revision_hash,
                   target_zh_tw, status, reviewer_id, provenance_json
            FROM translation_memory
            WHERE job_id=? AND chapter_id=? AND source_nfc=?
            ORDER BY updated_at DESC, memory_key
            """,
            (self.job_id, self.chapter_id, source_nfc),
        )
        return tuple(
            MemoryMatch(
                key=MemoryKey(str(row[0]), source_nfc, str(row[1]), str(row[2]), str(row[3])),
                target_zh_tw=str(row[4]),
                status=str(row[5]),  # type: ignore[arg-type]
                reviewer_id=str(row[6]) if row[6] is not None else None,
                provenance=json.loads(str(row[7])),
            )
            for row in rows
        )
