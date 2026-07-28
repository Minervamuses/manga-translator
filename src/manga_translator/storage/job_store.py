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
from ..domain.serialization import canonical_document_bytes, parse_document
from .artifact_store import ArtifactIntegrityError, ArtifactStore, require_local_storage

SCHEMA_VERSION = 2


class NewerDatabaseSchemaError(RuntimeError):
    pass


class MissingArtifactError(RuntimeError):
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
            script = resource.read_text(encoding="utf-8")
            with self.transaction() as connection:
                if not self._migration_already_applied(connection, target):
                    for statement in self._migration_statements(script):
                        connection.execute(statement)
                self._set_user_version(connection, target)

    @staticmethod
    def _migration_statements(script: str) -> tuple[str, ...]:
        statements: list[str] = []
        pending = ""
        for line in script.splitlines(keepends=True):
            pending += line
            if sqlite3.complete_statement(pending):
                statement = pending.strip()
                if statement:
                    statements.append(statement)
                pending = ""
        if pending.strip():
            raise RuntimeError("migration contains an incomplete SQL statement")
        return tuple(statements)

    @staticmethod
    def _migration_already_applied(
        connection: sqlite3.Connection, target: int
    ) -> bool:
        if target != 2:
            return False
        columns = {
            str(row[1]): row for row in connection.execute("PRAGMA table_info(stage_runs)")
        }
        cache_hits = columns.get("cache_hits")
        last_cache_hit_at = columns.get("last_cache_hit_at")
        return bool(
            cache_hits is not None
            and str(cache_hits[2]).upper() == "INTEGER"
            and int(cache_hits[3]) == 1
            and str(cache_hits[4]) == "0"
            and last_cache_hit_at is not None
            and str(last_cache_hit_at[2]).upper() == "TEXT"
            and int(last_cache_hit_at[3]) == 0
        )

    @staticmethod
    def _set_user_version(connection: sqlite3.Connection, target: int) -> None:
        connection.execute(f"PRAGMA user_version={target}")

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

    def ensure_job(self, job_id: str, *, config: dict[str, object] | None = None) -> None:
        config_json = json.dumps(
            config or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, status, config_json) VALUES (?, 'pending', ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    config_json=excluded.config_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
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
        if not self.artifacts.exists(artifact.sha256, expected_size=artifact.size_bytes):
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

    @staticmethod
    def _page_document_members(document: PageDocument) -> dict[str, int | None]:
        members: dict[str, int | None] = {}
        references = (
            document.source.original_artifact,
            *(ref for revision in document.region_revisions for ref in revision.mask_refs),
            *(record.raw_response_ref for record in document.translations),
            *(plan.font_ref for plan in document.layout_plans),
            *(plan.alpha_mask_ref for plan in document.layout_plans),
        )
        for reference in references:
            prior_size = members.setdefault(reference.sha256, reference.size_bytes)
            if prior_size != reference.size_bytes:
                raise ArtifactIntegrityError(
                    f"PageDocument has conflicting sizes for artifact {reference.sha256}"
                )
        for stage in document.stages:
            for sha256 in stage.output_hashes:
                members.setdefault(sha256, None)
        return members

    def _require_page_document_members(
        self, connection: sqlite3.Connection, document: PageDocument
    ) -> tuple[str, ...]:
        members = self._page_document_members(document)
        for sha256, expected_size in sorted(members.items()):
            row = connection.execute(
                "SELECT size_bytes FROM artifacts WHERE sha256=?", (sha256,)
            ).fetchone()
            if row is None:
                raise MissingArtifactError(
                    f"PageDocument member artifact {sha256} is not registered"
                )
            registered_size = int(row[0])
            if expected_size is not None and expected_size != registered_size:
                raise ArtifactIntegrityError(
                    f"PageDocument member artifact {sha256} records size "
                    f"{expected_size}, database has {registered_size}"
                )
            if not self.artifacts.exists(sha256, expected_size=registered_size):
                raise MissingArtifactError(
                    f"PageDocument member artifact {sha256} has no durable bytes"
                )
        return tuple(sorted(members))

    def store_page_document(self, job_id: str, document: PageDocument) -> ArtifactRef:
        data = canonical_document_bytes(document)
        artifact = self.artifacts.put_bytes(data, media_type="application/json")
        if not self.artifacts.exists(artifact.sha256, expected_size=artifact.size_bytes):
            raise RuntimeError("PageDocument artifact did not become durable")
        owner_id = f"{job_id}:{document.source.page_id}:document"
        with self.transaction() as connection:
            member_hashes = self._require_page_document_members(connection, document)
            self._insert_artifact(connection, artifact)
            connection.execute(
                "DELETE FROM artifact_references WHERE owner_type='page_document' AND owner_id=?",
                (owner_id,),
            )
            connection.execute(
                """
                DELETE FROM artifact_references
                WHERE owner_type='page_document_member' AND owner_id=?
                """,
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
            for member_hash in member_hashes:
                connection.execute(
                    """
                    INSERT INTO artifact_references(owner_type, owner_id, sha256)
                    VALUES ('page_document_member', ?, ?)
                    """,
                    (owner_id, member_hash),
                )
        return artifact

    def load_page_document(self, *, job_id: str, page_id: str) -> PageDocument | None:
        row = self.connection.execute(
            """
            SELECT document_artifact_sha256 FROM pages WHERE job_id=? AND page_id=?
            """,
            (job_id, page_id),
        ).fetchone()
        if row is None:
            return None
        return parse_document(self.artifacts.read_bytes(str(row[0])))

    def list_pages(self, *, job_id: str, page_id: str | None = None) -> list[sqlite3.Row]:
        if page_id is None:
            cursor = self.connection.execute(
                """
                SELECT page_id, document_artifact_sha256, created_at, updated_at
                FROM pages WHERE job_id=? ORDER BY page_id
                """,
                (job_id,),
            )
        else:
            cursor = self.connection.execute(
                """
                SELECT page_id, document_artifact_sha256, created_at, updated_at
                FROM pages WHERE job_id=? AND page_id=?
                """,
                (job_id, page_id),
            )
        return list(cursor)

    def list_stage_runs(self, *, job_id: str, page_id: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT stage, fingerprint, status, input_hashes_json, output_hashes_json,
                       started_at, finished_at, cache_hits, last_cache_hit_at
                FROM stage_runs
                WHERE job_id=? AND page_id=?
                ORDER BY stage_run_id
                """,
                (job_id, page_id),
            )
        )

    def referenced_hashes(self) -> set[str]:
        rows = self.connection.execute("SELECT DISTINCT sha256 FROM artifact_references")
        return {str(row[0]) for row in rows}

    def find_artifact(self, *, owner_type: str, owner_id: str) -> ArtifactRef | None:
        row = self.connection.execute(
            """
            SELECT artifacts.sha256, artifacts.media_type, artifacts.size_bytes
            FROM artifact_references
            JOIN artifacts USING(sha256)
            WHERE owner_type=? AND owner_id=?
            ORDER BY artifacts.sha256
            LIMIT 1
            """,
            (owner_type, owner_id),
        ).fetchone()
        if row is None:
            return None
        artifact = ArtifactRef(
            sha256=str(row[0]), media_type=str(row[1]), size_bytes=int(row[2])
        )
        return artifact if self.artifacts.exists(artifact.sha256) else None

    def interrupt_stale_stage_runs(self, *, job_id: str, page_id: str) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET status='interrupted', finished_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND page_id=? AND status='running'
                """,
                (job_id, page_id),
            )
        return cursor.rowcount

    def invalidate_stages(self, *, job_id: str, page_id: str, stages: set[str]) -> int:
        if not stages:
            return 0
        placeholders = ",".join("?" for _ in stages)
        with self.transaction() as connection:
            stale = list(
                connection.execute(
                    f"""
                    SELECT stage, fingerprint FROM stage_runs
                    WHERE job_id=? AND page_id=? AND stage IN ({placeholders})
                    """,
                    (job_id, page_id, *sorted(stages)),
                )
            )
            cursor = connection.execute(
                f"""
                UPDATE stage_runs SET status='invalidated', finished_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND page_id=? AND stage IN ({placeholders})
                """,
                (job_id, page_id, *sorted(stages)),
            )
            for stage, fingerprint in stale:
                prefix = f"{job_id}:{page_id}:{stage}:{fingerprint}:"
                connection.execute(
                    """
                    DELETE FROM artifact_references
                    WHERE owner_type='stage_output' AND substr(owner_id, 1, ?) = ?
                    """,
                    (len(prefix), prefix),
                )
        return cursor.rowcount

    def cached_stage_outputs(
        self, *, job_id: str, page_id: str, stage: str, fingerprint: str
    ) -> tuple[ArtifactRef, ...] | None:
        row = self.connection.execute(
            """
            SELECT output_hashes_json FROM stage_runs
            WHERE job_id=? AND page_id=? AND stage=? AND fingerprint=? AND status='succeeded'
            """,
            (job_id, page_id, stage, fingerprint),
        ).fetchone()
        if row is None:
            return None
        hashes = json.loads(str(row[0]))
        outputs: list[ArtifactRef] = []
        for sha256 in hashes:
            artifact_row = self.connection.execute(
                "SELECT media_type, size_bytes FROM artifacts WHERE sha256=?", (sha256,)
            ).fetchone()
            if artifact_row is None or not self.artifacts.exists(sha256):
                with self.transaction() as connection:
                    connection.execute(
                        """
                        UPDATE stage_runs SET status='invalidated', finished_at=CURRENT_TIMESTAMP
                        WHERE job_id=? AND page_id=? AND stage=? AND fingerprint=?
                        """,
                        (job_id, page_id, stage, fingerprint),
                    )
                return None
            outputs.append(
                ArtifactRef(
                    sha256=sha256,
                    media_type=str(artifact_row[0]),
                    size_bytes=int(artifact_row[1]),
                )
            )
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE stage_runs
                SET cache_hits=cache_hits + 1, last_cache_hit_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND page_id=? AND stage=? AND fingerprint=?
                """,
                (job_id, page_id, stage, fingerprint),
            )
        return tuple(outputs)

    def start_stage(
        self,
        *,
        job_id: str,
        page_id: str,
        stage: str,
        fingerprint: str,
        input_hashes: tuple[str, ...],
    ) -> None:
        with self.transaction() as connection:
            prefix = f"{job_id}:{page_id}:{stage}:{fingerprint}:"
            connection.execute(
                """
                DELETE FROM artifact_references
                WHERE owner_type='stage_output' AND substr(owner_id, 1, ?) = ?
                """,
                (len(prefix), prefix),
            )
            connection.execute(
                """
                INSERT INTO stage_runs(
                    job_id, page_id, stage, fingerprint, status, input_hashes_json, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(job_id, page_id, stage, fingerprint) DO UPDATE SET
                    status='running',
                    input_hashes_json=excluded.input_hashes_json,
                    output_hashes_json='[]',
                    started_at=CURRENT_TIMESTAMP,
                    finished_at=NULL
                """,
                (job_id, page_id, stage, fingerprint, json.dumps(input_hashes)),
            )

    def finish_stage(
        self,
        *,
        job_id: str,
        page_id: str,
        stage: str,
        fingerprint: str,
        output_hashes: tuple[str, ...],
    ) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET status='succeeded', output_hashes_json=?, finished_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND page_id=? AND stage=? AND fingerprint=? AND status='running'
                """,
                (json.dumps(output_hashes), job_id, page_id, stage, fingerprint),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("running stage record disappeared before completion")

    def fail_stage(
        self, *, job_id: str, page_id: str, stage: str, fingerprint: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE stage_runs SET status='failed', finished_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND page_id=? AND stage=? AND fingerprint=?
                """,
                (job_id, page_id, stage, fingerprint),
            )

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
