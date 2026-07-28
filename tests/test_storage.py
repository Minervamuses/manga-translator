from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

import pytest

from manga_translator.domain.ids import page_id_from_bytes
from manga_translator.domain.models import ArtifactRef, PageDocument, SourcePage
from manga_translator.storage.artifact_store import (
    ArtifactIntegrityError,
    ArtifactStore,
    NetworkShareError,
    assess_storage_path,
    require_local_storage,
)
from manga_translator.storage.job_store import (
    JobStore,
    MissingArtifactError,
    NewerDatabaseSchemaError,
)


def _document(original_artifact: ArtifactRef | None = None) -> PageDocument:
    page_id = page_id_from_bytes(b"source")
    return PageDocument(
        source=SourcePage(
            page_id=page_id,
            original_bytes_sha256=page_id,
            source_path="page.png",
            width=10,
            height=10,
            mode="RGB",
            original_artifact=original_artifact
            or ArtifactRef(sha256=page_id, media_type="image/png", size_bytes=6),
        )
    )


def _store_source(jobs: JobStore) -> ArtifactRef:
    return jobs.store_artifact(
        b"source", media_type="image/png", owner_type="source", owner_id="page"
    )


def test_same_bytes_create_one_atomic_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes(b"same bytes", media_type="application/octet-stream")
    second = store.put_bytes(b"same bytes", media_type="application/octet-stream")

    assert first == second
    assert store.read_bytes(first.sha256) == b"same bytes"
    assert list(store.iter_hashes()) == [first.sha256]
    assert not list(store.root.rglob("*.tmp"))


def test_existing_artifact_is_revalidated_before_reuse(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes(b"original", media_type="application/octet-stream")
    store.path_for(artifact.sha256).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="content hashes to"):
        store.put_bytes(b"original", media_type="application/octet-stream")


def test_read_and_exists_fail_closed_on_corrupt_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes(b"original", media_type="application/octet-stream")
    store.path_for(artifact.sha256).write_bytes(b"short")

    with pytest.raises(ArtifactIntegrityError, match="has size 5, expected 8"):
        store.exists(artifact.sha256, expected_size=artifact.size_bytes)
    with pytest.raises(ArtifactIntegrityError, match="has size 5, expected 8"):
        store.read_bytes(artifact.sha256, expected_size=artifact.size_bytes)


def test_database_migration_is_idempotent_and_enables_foreign_keys(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(database, artifacts) as first:
        tables = {
            row[0]
            for row in first.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "jobs",
            "pages",
            "stage_runs",
            "region_identities",
            "region_revisions",
            "artifacts",
            "issues",
            "entities",
        } <= tables
        assert first.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert first.connection.execute("PRAGMA user_version").fetchone()[0] == 2

    with JobStore(database, artifacts) as reopened:
        assert reopened.connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_migration_retries_committed_ddl_without_user_version_bump(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(database, artifacts):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")

    with JobStore(database, artifacts) as recovered:
        columns = {
            row[1] for row in recovered.connection.execute("PRAGMA table_info(stage_runs)")
        }
        assert {"cache_hits", "last_cache_hit_at"} <= columns
        assert recovered.connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_migration_ddl_and_version_bump_are_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "jobs.sqlite3"
    initial = files("manga_translator.storage.migrations").joinpath("001_initial.sql")
    with sqlite3.connect(database) as connection:
        connection.executescript(initial.read_text(encoding="utf-8"))
        connection.execute("PRAGMA user_version=1")

    def fail_version_bump(_connection: sqlite3.Connection, _target: int) -> None:
        raise RuntimeError("simulated crash before version bump")

    monkeypatch.setattr(JobStore, "_set_user_version", staticmethod(fail_version_bump))
    with pytest.raises(RuntimeError, match="simulated crash"):
        JobStore(database, ArtifactStore(tmp_path / "artifacts"))

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(stage_runs)")}
        assert "cache_hits" not in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

    monkeypatch.undo()
    with JobStore(database, ArtifactStore(tmp_path / "artifacts")) as recovered:
        assert recovered.connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_newer_database_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "new.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(NewerDatabaseSchemaError):
        JobStore(database, ArtifactStore(tmp_path / "artifacts"))


def test_page_reference_is_committed_only_after_artifact_is_durable(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(tmp_path / "jobs.sqlite3", artifacts) as jobs:
        jobs.create_job("job-1")
        artifact = jobs.store_page_document("job-1", _document(_store_source(jobs)))
        row = jobs.connection.execute(
            "SELECT document_artifact_sha256 FROM pages WHERE job_id='job-1'"
        ).fetchone()

        assert row[0] == artifact.sha256
        assert artifacts.exists(row[0])
        with pytest.raises(sqlite3.IntegrityError):
            jobs.connection.execute(
                "UPDATE pages SET document_artifact_sha256=? WHERE job_id='job-1'",
                ("0" * 64,),
            )


def test_artifact_write_failure_cannot_create_database_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(tmp_path / "jobs.sqlite3", artifacts) as jobs:
        jobs.create_job("job-1")
        source = _store_source(jobs)

        def fail_write(*_args: object, **_kwargs: object) -> ArtifactRef:
            raise OSError("simulated interruption")

        monkeypatch.setattr(artifacts, "put_bytes", fail_write)
        with pytest.raises(OSError, match="simulated interruption"):
            jobs.store_page_document("job-1", _document(source))
        assert jobs.connection.execute("SELECT count(*) FROM pages").fetchone()[0] == 0
        assert (
            jobs.connection.execute("SELECT count(*) FROM artifact_references").fetchone()[0]
            == 1
        )


def test_page_document_rejects_unregistered_member_artifact(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(tmp_path / "jobs.sqlite3", artifacts) as jobs:
        jobs.create_job("job-1")

        with pytest.raises(MissingArtifactError, match="is not registered"):
            jobs.store_page_document("job-1", _document())

        assert jobs.connection.execute("SELECT count(*) FROM pages").fetchone()[0] == 0


def test_page_document_rejects_corrupt_member_artifact(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with JobStore(tmp_path / "jobs.sqlite3", artifacts) as jobs:
        jobs.create_job("job-1")
        source = _store_source(jobs)
        artifacts.path_for(source.sha256).write_bytes(b"bad")

        with pytest.raises(ArtifactIntegrityError, match="has size 3, expected 6"):
            jobs.store_page_document("job-1", _document(source))

        assert jobs.connection.execute("SELECT count(*) FROM pages").fetchone()[0] == 0


def test_gc_removes_unreferenced_records_and_orphan_files(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    orphan = artifacts.put_bytes(b"orphan", media_type="text/plain")
    with JobStore(tmp_path / "jobs.sqlite3", artifacts) as jobs:
        jobs.create_job("job-1")
        kept = jobs.store_artifact(
            b"kept", media_type="text/plain", owner_type="test", owner_id="one"
        )
        registered_orphan = jobs.store_artifact(
            b"registered orphan", media_type="text/plain", owner_type="test", owner_id="two"
        )
        jobs.connection.execute(
            "DELETE FROM artifact_references WHERE sha256=?", (registered_orphan.sha256,)
        )
        result = jobs.gc()

        assert result.database_records == 1
        assert result.files == 2
        assert artifacts.exists(kept.sha256)
        assert not artifacts.exists(orphan.sha256)
        assert not artifacts.exists(registered_orphan.sha256)


def test_explicit_network_share_is_rejected() -> None:
    assessment = assess_storage_path(r"\\server\share\manga-cache")
    assert assessment.kind == "network"
    with pytest.raises(NetworkShareError):
        require_local_storage(r"\\server\share\manga-cache")
