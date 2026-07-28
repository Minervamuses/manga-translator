"""Approval-aware translation memory with context and entity revision binding."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from ..storage.job_store import JobStore

MemoryStatus = Literal["suggestion", "approved"]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        return self.status == "approved" and self.reviewer_id is not None


def build_memory_key(
    source: str,
    *,
    context: Any,
    order: Any,
    entity_revision_hash: str,
) -> MemoryKey:
    source_nfc = unicodedata.normalize("NFC", source)
    if not source_nfc.strip():
        raise ValueError("translation memory source must not be empty")
    if len(entity_revision_hash) != 64:
        raise ValueError("entity_revision_hash must be SHA-256")
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


class TranslationMemory:
    def __init__(self, store: JobStore, *, job_id: str, chapter_id: str) -> None:
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
        target = unicodedata.normalize("NFC", target_zh_tw).strip()
        if not target:
            raise ValueError("translation memory target must not be empty")
        if status == "approved" and not (reviewer_id or "").strip():
            raise ValueError("approved translation memory requires reviewer_id")
        payload = json.dumps(
            provenance or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
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
                    reviewer_id,
                    payload,
                ),
            )
        match = self.lookup_key(key)
        assert match is not None
        return match

    def lookup_key(self, key: MemoryKey) -> MemoryMatch | None:
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
        source_nfc = unicodedata.normalize("NFC", source)
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
