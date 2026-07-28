"""SQLite metadata store whose references only target durable artifacts."""

from __future__ import annotations

import json
import sqlite3
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from ..domain.models import ArtifactRef, PageDocument
from ..domain.serialization import canonical_document_bytes
from .artifact_store import ArtifactStore, require_local_storage

SCHEMA_VERSION = 1


class NewerDatabaseSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class GarbageCollectionResult:
    database_records: int
    files: int


class JobStore:
    def __init__(self, database: str | Path, artifacts: ArtifactStore) -> None:
        self.database = Path(database).expanduser().resolve(strict=False)
        self.path_assessment = require_local_storage(self.database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifacts
        self.connection = sqlite3.connect(self.database, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            self._configure()
            self._migrate()
        except BaseException:
            self.connection.close()
            raise

    def _configure(self) -> None:
        self.connection.execute("PRAGMA foreign_keys=ON")
        enabled = self.connection.execute("PRAGMA foreign_keys").fetchone()[0]
        if enabled != 1:
            raise RuntimeError("SQLite foreign key enforcement could not be enabled")
        mode = str(self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if mode != "wal":
            warnings.warn(
                f"SQLite WAL unavailable (journal_mode={mode}); using rollback journal",
                RuntimeWarning,
                stacklevel=2,
            )
            self.connection.execute("PRAGMA journal_mode=DELETE")

    def _migrate(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise NewerDatabaseSchemaError(
                f"database schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        migration_root = files("manga_translator.storage.migrations")
        for target in range(version + 1, SCHEMA_VERSION + 1):
            resource = migration_root.joinpath(f"{target:03d}_initial.sql")
            self.connection.executescript(resource.read_text(encoding="utf-8"))
            self.connection.execute(f"PRAGMA user_version={target}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create_job(self, job_id: str, *, config: dict[str, object] | None = None) -> None:
        config_json = json.dumps(
            config or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(job_id, status, config_json) VALUES (?, 'pending', ?)",
                (job_id, config_json),
            )

    @staticmethod
    def _insert_artifact(connection: sqlite3.Connection, artifact: ArtifactRef) -> None:
        connection.execute(
            """
            INSERT INTO artifacts(sha256, media_type, size_bytes)
            VALUES (?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                media_type=excluded.media_type,
                size_bytes=excluded.size_bytes
            """,
            (artifact.sha256, artifact.media_type, artifact.size_bytes),
        )

    def store_artifact(
        self,
        data: bytes,
        *,
        media_type: str,
        owner_type: str,
        owner_id: str,
    ) -> ArtifactRef:
        artifact = self.artifacts.put_bytes(data, media_type=media_type)
        if not self.artifacts.exists(artifact.sha256):
            raise RuntimeError("artifact did not become durable")
        with self.transaction() as connection:
            self._insert_artifact(connection, artifact)
            connection.execute(
                """
                INSERT OR IGNORE INTO artifact_references(owner_type, owner_id, sha256)
                VALUES (?, ?, ?)
                """,
                (owner_type, owner_id, artifact.sha256),
            )
        return artifact

    def store_page_document(self, job_id: str, document: PageDocument) -> ArtifactRef:
        data = canonical_document_bytes(document)
        artifact = self.artifacts.put_bytes(data, media_type="application/json")
        if not self.artifacts.exists(artifact.sha256):
            raise RuntimeError("PageDocument artifact did not become durable")
        owner_id = f"{job_id}:{document.source.page_id}:document"
        with self.transaction() as connection:
            self._insert_artifact(connection, artifact)
            connection.execute(
                "DELETE FROM artifact_references WHERE owner_type='page_document' AND owner_id=?",
                (owner_id,),
            )
            connection.execute(
                """
                INSERT INTO pages(job_id, page_id, document_artifact_sha256)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id, page_id) DO UPDATE SET
                    document_artifact_sha256=excluded.document_artifact_sha256,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (job_id, document.source.page_id, artifact.sha256),
            )
            connection.execute(
                "INSERT INTO artifact_references(owner_type, owner_id, sha256) VALUES (?, ?, ?)",
                ("page_document", owner_id, artifact.sha256),
            )
        return artifact

    def referenced_hashes(self) -> set[str]:
        rows = self.connection.execute("SELECT DISTINCT sha256 FROM artifact_references")
        return {str(row[0]) for row in rows}

    def gc(self) -> GarbageCollectionResult:
        referenced = self.referenced_hashes()
        registered = {
            str(row[0]) for row in self.connection.execute("SELECT sha256 FROM artifacts")
        }
        unreferenced = registered - referenced
        with self.transaction() as connection:
            for sha256 in unreferenced:
                connection.execute("DELETE FROM artifacts WHERE sha256=?", (sha256,))
        removed_files = 0
        for sha256 in set(self.artifacts.iter_hashes()) - referenced:
            removed_files += int(self.artifacts.delete(sha256))
        return GarbageCollectionResult(len(unreferenced), removed_files)
