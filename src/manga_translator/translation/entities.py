"""Chapter-scoped entity ledger with explicit human approval boundaries."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..storage.job_store import JobStore

EntityStatus = Literal["candidate", "approved", "rejected", "merged"]
_ACTIVE_ENTITY_STATUSES = frozenset({"candidate", "approved"})


def normalize_entity_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("entity text must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        raise ValueError("entity text must not be empty")
    return normalized


def _nonempty_label(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _json_object(value: dict[str, Any], field: str) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    canonical_source: str
    aliases: tuple[str, ...]
    approved_zh_tw: str | None
    kind: str
    scope: str
    status: EntityStatus
    provenance: dict[str, Any]
    first_seen: str
    last_seen: str
    merged_into: str | None


class EntityLedger:
    def __init__(self, store: JobStore, *, job_id: str, chapter_id: str) -> None:
        _nonempty_label(job_id, "job_id")
        _nonempty_label(chapter_id, "chapter_id")
        self.store = store
        self.job_id = job_id
        self.chapter_id = chapter_id

    def _entity_id(self, canonical_source: str) -> str:
        material = f"{self.job_id}\0{self.chapter_id}\0{canonical_source}".encode()
        return "entity-" + hashlib.sha256(material).hexdigest()[:20]

    def _aliases(self, entity_id: str) -> tuple[str, ...]:
        rows = self.store.connection.execute(
            """
            SELECT alias FROM entity_aliases
            WHERE job_id=? AND chapter_id=? AND entity_id=?
            ORDER BY normalized_alias
            """,
            (self.job_id, self.chapter_id, entity_id),
        )
        return tuple(str(row[0]) for row in rows)

    def _from_row(self, row) -> Entity:
        return Entity(
            entity_id=str(row[0]),
            canonical_source=str(row[1]),
            aliases=self._aliases(str(row[0])),
            approved_zh_tw=str(row[2]) if row[2] is not None else None,
            kind=str(row[3]),
            scope=str(row[4]),
            status=str(row[5]),  # type: ignore[arg-type]
            provenance=json.loads(str(row[6])),
            first_seen=str(row[7]),
            last_seen=str(row[8]),
            merged_into=str(row[9]) if row[9] is not None else None,
        )

    def get(self, entity_id: str) -> Entity | None:
        row = self.store.connection.execute(
            """
            SELECT entity_id, canonical_source, approved_zh_tw, kind, scope, status,
                   provenance_json, first_seen, last_seen, merged_into
            FROM chapter_entities WHERE job_id=? AND chapter_id=? AND entity_id=?
            """,
            (self.job_id, self.chapter_id, entity_id),
        ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(self, *, status: EntityStatus | None = None) -> tuple[Entity, ...]:
        parameters: list[str] = [self.job_id, self.chapter_id]
        where = "job_id=? AND chapter_id=?"
        if status is not None:
            where += " AND status=?"
            parameters.append(status)
        rows = self.store.connection.execute(
            f"""
            SELECT entity_id, canonical_source, approved_zh_tw, kind, scope, status,
                   provenance_json, first_seen, last_seen, merged_into
            FROM chapter_entities WHERE {where} ORDER BY canonical_source, entity_id
            """,
            parameters,
        )
        return tuple(self._from_row(row) for row in rows)

    def list_candidates(self) -> tuple[Entity, ...]:
        return self.list(status="candidate")

    def _find_by_alias(self, aliases: tuple[str, ...]) -> str | None:
        placeholders = ",".join("?" for _ in aliases)
        rows = self.store.connection.execute(
            f"""
            SELECT entity_id FROM entity_aliases
            WHERE job_id=? AND chapter_id=? AND normalized_alias IN ({placeholders})
            GROUP BY entity_id ORDER BY entity_id
            """,
            (self.job_id, self.chapter_id, *aliases),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError("aliases refer to multiple entities; a reviewed merge is required")
        return str(rows[0][0]) if rows else None

    def propose(
        self,
        canonical_source: str,
        *,
        aliases: tuple[str, ...] = (),
        kind: str = "unknown",
        scope: str = "chapter",
        provenance: dict[str, Any] | None = None,
    ) -> Entity:
        canonical = normalize_entity_text(canonical_source)
        normalized_kind = _nonempty_label(kind, "kind")
        normalized_scope = _nonempty_label(scope, "scope")
        originals = tuple(dict.fromkeys((canonical, *(normalize_entity_text(x) for x in aliases))))
        normalized_aliases = tuple(normalize_entity_text(item) for item in originals)
        existing_id = self._find_by_alias(normalized_aliases)
        entity_id = existing_id or self._entity_id(canonical)
        payload = _json_object(
            {"source": "model"} if provenance is None else provenance,
            "provenance",
        )
        with self.store.transaction() as connection:
            if existing_id is None:
                connection.execute(
                    """
                    INSERT INTO chapter_entities(
                        job_id, chapter_id, entity_id, canonical_source, approved_zh_tw,
                        kind, scope, status, provenance_json
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?, 'candidate', ?)
                    """,
                    (
                        self.job_id,
                        self.chapter_id,
                        entity_id,
                        canonical,
                        normalized_kind,
                        normalized_scope,
                        payload,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE chapter_entities SET last_seen=CURRENT_TIMESTAMP
                    WHERE job_id=? AND chapter_id=? AND entity_id=?
                    """,
                    (self.job_id, self.chapter_id, entity_id),
                )
            for alias, normalized in zip(originals, normalized_aliases, strict=True):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases(
                        job_id, chapter_id, entity_id, alias, normalized_alias, source
                    ) VALUES (?, ?, ?, ?, ?, 'ocr_or_model')
                    """,
                    (self.job_id, self.chapter_id, entity_id, alias, normalized),
                )
        result = self.get(entity_id)
        assert result is not None
        return result

    def approve(self, entity_id: str, approved_zh_tw: str, *, reviewer_id: str) -> Entity:
        target = normalize_entity_text(approved_zh_tw)
        reviewer = _nonempty_label(reviewer_id, "reviewer_id")
        with self.store.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE chapter_entities
                SET approved_zh_tw=?, status='approved', last_seen=CURRENT_TIMESTAMP,
                    provenance_json=json_set(provenance_json, '$.approval_reviewer', ?)
                WHERE job_id=? AND chapter_id=? AND entity_id=? AND status IN ('candidate','approved')
                """,
                (target, reviewer, self.job_id, self.chapter_id, entity_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"entity is not approvable: {entity_id}")
        result = self.get(entity_id)
        assert result is not None
        return result

    def reject(self, entity_id: str, *, reviewer_id: str) -> Entity:
        reviewer = _nonempty_label(reviewer_id, "reviewer_id")
        with self.store.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE chapter_entities SET status='rejected', approved_zh_tw=NULL,
                    last_seen=CURRENT_TIMESTAMP,
                    provenance_json=json_set(provenance_json, '$.rejection_reviewer', ?)
                WHERE job_id=? AND chapter_id=? AND entity_id=? AND status='candidate'
                """,
                (reviewer, self.job_id, self.chapter_id, entity_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"entity is not rejectable: {entity_id}")
        result = self.get(entity_id)
        assert result is not None
        return result

    def merge(self, source_id: str, target_id: str, *, reviewer_id: str) -> Entity:
        if source_id == target_id:
            raise ValueError("cannot merge an entity into itself")
        reviewer = _nonempty_label(reviewer_id, "reviewer_id")
        with self.store.transaction() as connection:
            rows = connection.execute(
                """
                SELECT entity_id, status, approved_zh_tw
                FROM chapter_entities
                WHERE job_id=? AND chapter_id=? AND entity_id IN (?, ?)
                """,
                (self.job_id, self.chapter_id, source_id, target_id),
            ).fetchall()
            by_id = {str(row[0]): row for row in rows}
            if source_id not in by_id or target_id not in by_id:
                raise KeyError("merge requires two existing entities")
            source_status = str(by_id[source_id][1])
            target_status = str(by_id[target_id][1])
            if (
                source_status not in _ACTIVE_ENTITY_STATUSES
                or target_status not in _ACTIVE_ENTITY_STATUSES
            ):
                raise ValueError("only active candidate or approved entities may be merged")
            source_target = by_id[source_id][2]
            target_target = by_id[target_id][2]
            if source_target is not None and target_target is not None and source_target != target_target:
                raise ValueError("approved entities with different translations cannot be merged")
            if source_target is not None and target_target is None:
                connection.execute(
                    """
                    UPDATE chapter_entities
                    SET approved_zh_tw=?, status='approved', last_seen=CURRENT_TIMESTAMP,
                        provenance_json=json_set(provenance_json, '$.merge_approval_reviewer', ?)
                    WHERE job_id=? AND chapter_id=? AND entity_id=?
                    """,
                    (source_target, reviewer, self.job_id, self.chapter_id, target_id),
                )
            aliases = list(
                connection.execute(
                    """
                    SELECT alias, normalized_alias, source FROM entity_aliases
                    WHERE job_id=? AND chapter_id=? AND entity_id=?
                    """,
                    (self.job_id, self.chapter_id, source_id),
                )
            )
            connection.execute(
                """
                DELETE FROM entity_aliases
                WHERE job_id=? AND chapter_id=? AND entity_id=?
                """,
                (self.job_id, self.chapter_id, source_id),
            )
            for alias, normalized, source in aliases:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO entity_aliases(
                        job_id, chapter_id, entity_id, alias, normalized_alias, source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (self.job_id, self.chapter_id, target_id, alias, normalized, source),
                )
            cursor = connection.execute(
                """
                UPDATE chapter_entities SET status='merged', approved_zh_tw=NULL,
                    merged_into=?, last_seen=CURRENT_TIMESTAMP,
                    provenance_json=json_set(provenance_json, '$.merge_reviewer', ?)
                WHERE job_id=? AND chapter_id=? AND entity_id=?
                """,
                (target_id, reviewer, self.job_id, self.chapter_id, source_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"entity is not mergeable: {source_id}")
        result = self.get(target_id)
        assert result is not None
        return result

    def import_glossary(self, glossary: str | Path | dict[str, str]) -> tuple[Entity, ...]:
        if isinstance(glossary, (str, Path)):
            raw = json.loads(Path(glossary).read_text(encoding="utf-8"))
            entries = raw.get("entries", raw) if isinstance(raw, dict) else None
        else:
            entries = glossary
        if not isinstance(entries, dict):
            raise TypeError("glossary must be a JSON object or contain an entries object")
        normalized_entries: list[tuple[str, str]] = []
        seen: dict[str, str] = {}
        for source, target in entries.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise TypeError("glossary entries must map strings to strings")
            normalized_source = normalize_entity_text(source)
            normalized_target = normalize_entity_text(target)
            previous = seen.get(normalized_source)
            if previous is not None and previous != normalized_target:
                raise ValueError("normalized glossary source has conflicting translations")
            seen[normalized_source] = normalized_target
            normalized_entries.append((normalized_source, normalized_target))
        imported = []
        for source, target in normalized_entries:
            candidate = self.propose(
                source, provenance={"source": "glossary_import", "trusted": True}
            )
            imported.append(
                self.approve(candidate.entity_id, target, reviewer_id="glossary_import")
            )
        return tuple(imported)

    def export_glossary(self, path: str | Path | None = None) -> dict[str, dict[str, str]]:
        payload = {
            "entries": {
                entity.canonical_source: entity.approved_zh_tw
                for entity in self.list(status="approved")
                if entity.approved_zh_tw is not None
            }
        }
        if path is not None:
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return payload

    def approved_constraints(self) -> dict[str, str]:
        return {
            alias: entity.approved_zh_tw
            for entity in self.list(status="approved")
            if entity.approved_zh_tw is not None
            for alias in entity.aliases
        }

    def prompt_entries(self, *, include_candidates: bool = True) -> tuple[dict[str, Any], ...]:
        entries: list[dict[str, Any]] = [
            {
                "source": entity.canonical_source,
                "target": entity.approved_zh_tw,
                "status": "approved",
                "constraint": "hard",
            }
            for entity in self.list(status="approved")
        ]
        if include_candidates:
            entries.extend(
                {
                    "source": entity.canonical_source,
                    "target": None,
                    "status": "candidate",
                    "constraint": "hint",
                }
                for entity in self.list(status="candidate")
            )
        return tuple(entries)

    def revision_hash(self) -> str:
        material = [
            {
                "entity_id": entity.entity_id,
                "canonical_source": entity.canonical_source,
                "aliases": entity.aliases,
                "approved_zh_tw": entity.approved_zh_tw,
                "kind": entity.kind,
                "scope": entity.scope,
                "status": entity.status,
                "merged_into": entity.merged_into,
            }
            for entity in self.list()
        ]
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
