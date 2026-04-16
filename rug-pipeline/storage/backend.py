"""Storage backend protocol and factory.

Defines the interface that all storage backends must implement,
and provides a factory function to create the appropriate backend
based on configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends (local filesystem, S3, etc.)."""

    def put(self, local_path: Path, remote_key: str, bucket: str) -> None:
        """Upload a local file to the storage backend.

        Args:
            local_path: Path to the local file to upload.
            remote_key: Destination key/path in the backend.
            bucket: Bucket name (S3) or base directory (local).
        """
        ...

    def get(self, remote_key: str, bucket: str, local_path: Path) -> Path:
        """Download a file from the storage backend to a local path.

        Args:
            remote_key: Source key/path in the backend.
            bucket: Bucket name (S3) or base directory (local).
            local_path: Destination path on local filesystem.

        Returns:
            The local_path where the file was downloaded.
        """
        ...

    def exists(self, remote_key: str, bucket: str) -> bool:
        """Check if a key exists in the storage backend."""
        ...

    def delete(self, remote_key: str, bucket: str) -> None:
        """Delete a key from the storage backend."""
        ...

    def list_keys(self, prefix: str, bucket: str) -> list[str]:
        """List all keys under a prefix."""
        ...

    def read_json(self, remote_key: str, bucket: str) -> dict:
        """Read and parse a JSON file from the backend."""
        ...

    def write_json(self, data: dict, remote_key: str, bucket: str) -> None:
        """Write a dict as JSON to the backend."""
        ...


def create_backend(backend_type: str, **kwargs) -> StorageBackend:
    """Factory function to create a storage backend.

    Args:
        backend_type: Either "local" or "s3".
        **kwargs: Backend-specific configuration.

    Returns:
        A StorageBackend implementation.
    """
    if backend_type == "local":
        from storage.local import LocalBackend
        return LocalBackend()
    elif backend_type == "s3":
        from storage.s3 import S3Backend
        return S3Backend(
            region=kwargs.get("region", "us-east-1"),
            glacier_transition_days=kwargs.get("glacier_transition_days", 1),
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_type!r}. Use 'local' or 's3'.")
