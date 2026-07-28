"""Atomic content-addressed artifact storage."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..domain.models import ArtifactRef


class NetworkShareError(ValueError):
    pass


class ArtifactIntegrityError(RuntimeError):
    """Raised when content-addressed bytes do not match their recorded identity."""


@dataclass(frozen=True)
class StoragePathAssessment:
    kind: Literal["local", "network", "unknown"]
    reason: str


def assess_storage_path(path: str | Path) -> StoragePathAssessment:
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")):
        return StoragePathAssessment("network", "UNC paths are not supported for durable state")
    resolved = Path(path).expanduser().resolve(strict=False)
    if os.name != "nt":
        return StoragePathAssessment("local", "non-Windows path without an explicit network prefix")
    try:
        import ctypes

        root = resolved.anchor
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(root)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return StoragePathAssessment("unknown", "Windows drive type could not be determined")
    if drive_type == 4:
        return StoragePathAssessment("network", f"{root} is a mapped network drive")
    if drive_type in {0, 1}:
        return StoragePathAssessment("unknown", f"Windows returned drive type {drive_type}")
    return StoragePathAssessment("local", f"Windows drive type {drive_type}")


def require_local_storage(path: str | Path) -> StoragePathAssessment:
    assessment = assess_storage_path(path)
    if assessment.kind == "network":
        raise NetworkShareError(assessment.reason)
    return assessment


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.path_assessment = require_local_storage(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_sha256(sha256: str) -> None:
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")

    def path_for(self, sha256: str) -> Path:
        self._validate_sha256(sha256)
        return self.root / sha256[:2] / sha256[2:]

    @staticmethod
    def _verify_bytes(
        data: bytes, *, sha256: str, expected_size: int | None = None
    ) -> None:
        if expected_size is not None and len(data) != expected_size:
            raise ArtifactIntegrityError(
                f"artifact {sha256} has size {len(data)}, expected {expected_size}"
            )
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != sha256:
            raise ArtifactIntegrityError(
                f"artifact {sha256} content hashes to {actual_sha256}"
            )

    def verify(self, sha256: str, *, expected_size: int | None = None) -> bool:
        """Return whether an artifact exists, raising if existing bytes are corrupt."""

        path = self.path_for(sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return False
        self._verify_bytes(data, sha256=sha256, expected_size=expected_size)
        return True

    def put_bytes(self, data: bytes, *, media_type: str) -> ArtifactRef:
        sha256 = hashlib.sha256(data).hexdigest()
        destination = self.path_for(sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.verify(sha256, expected_size=len(data)):
            return ArtifactRef(sha256=sha256, media_type=media_type, size_bytes=len(data))
        if not destination.exists():
            temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if destination.exists():
                    temporary.unlink()
                else:
                    os.replace(temporary, destination)
                    self._fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        if not self.verify(sha256, expected_size=len(data)):
            raise RuntimeError("artifact did not become durable")
        return ArtifactRef(sha256=sha256, media_type=media_type, size_bytes=len(data))

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_bytes(self, sha256: str, *, expected_size: int | None = None) -> bytes:
        data = self.path_for(sha256).read_bytes()
        self._verify_bytes(data, sha256=sha256, expected_size=expected_size)
        return data

    def exists(self, sha256: str, *, expected_size: int | None = None) -> bool:
        return self.verify(sha256, expected_size=expected_size)

    def delete(self, sha256: str) -> bool:
        path = self.path_for(sha256)
        if not path.exists():
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def iter_hashes(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for artifact in sorted(shard.iterdir()):
                candidate = shard.name + artifact.name
                try:
                    self._validate_sha256(candidate)
                except ValueError:
                    continue
                if artifact.is_file():
                    yield candidate
