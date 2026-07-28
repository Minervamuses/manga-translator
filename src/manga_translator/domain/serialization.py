"""Strict canonical serialization and explicit schema migration entrypoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from .models import SCHEMA_VERSION, PageDocument


class DocumentSchemaError(ValueError):
    pass


class UnsupportedSchemaVersion(DocumentSchemaError):
    pass


class MigrationRequired(DocumentSchemaError):
    pass


def canonical_document_bytes(document: PageDocument) -> bytes:
    payload = document.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, ValueError) as error:
        raise DocumentSchemaError(f"invalid schema_version: {version!r}") from error


def parse_document(raw: bytes | str) -> PageDocument:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DocumentSchemaError("invalid PageDocument JSON") from error
    if not isinstance(payload, dict):
        raise DocumentSchemaError("PageDocument root must be an object")
    version = payload.get("schema_version")
    if version is None:
        raise DocumentSchemaError("schema_version is required")
    parsed = _version_tuple(version)
    current = _version_tuple(SCHEMA_VERSION)
    if parsed > current:
        raise UnsupportedSchemaVersion(f"newer schema_version is unsupported: {version}")
    if parsed < current:
        raise MigrationRequired(f"schema_version {version} requires explicit migration")
    try:
        return PageDocument.model_validate_json(raw)
    except ValidationError as error:
        raise DocumentSchemaError(str(error)) from error


Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _migrate_0_9_to_1_0(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


MIGRATIONS: dict[str, Migration] = {"0.9": _migrate_0_9_to_1_0}


def migrate_document(payload: dict[str, Any]) -> PageDocument:
    version = payload.get("schema_version")
    migration = MIGRATIONS.get(str(version))
    if migration is None:
        raise MigrationRequired(f"no explicit migration registered for schema_version {version}")
    migrated = migration(payload)
    try:
        raw = json.dumps(migrated, ensure_ascii=False, allow_nan=False)
        return PageDocument.model_validate_json(raw)
    except (TypeError, ValueError, ValidationError) as error:
        raise DocumentSchemaError(str(error)) from error
