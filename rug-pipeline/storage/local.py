"""Local filesystem storage backend.

Maps the StorageBackend protocol to local filesystem operations.
The 'bucket' parameter is treated as a base directory path.
"""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalBackend:
    """Storage backend using the local filesystem.

    In this backend, 'bucket' is a directory path on the local filesystem,
    and 'remote_key' is a relative path within that directory.
    """

    def put(self, local_path: Path, remote_key: str, bucket: str) -> None:
        """Copy a local file to the target location."""
        dest = Path(bucket) / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if local_path.resolve() == dest.resolve():
            logger.debug("put: source and dest are the same, skipping: %s", dest)
            return

        shutil.copy2(str(local_path), str(dest))
        logger.debug("put: %s → %s", local_path, dest)

    def get(self, remote_key: str, bucket: str, local_path: Path) -> Path:
        """Copy a file from the backend to a local path."""
        source = Path(bucket) / remote_key

        if not source.exists():
            raise FileNotFoundError(f"Key not found: {remote_key} in {bucket}")

        if source.resolve() == local_path.resolve():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(local_path))
        logger.debug("get: %s → %s", source, local_path)
        return local_path

    def exists(self, remote_key: str, bucket: str) -> bool:
        """Check if a file exists at the given key."""
        return (Path(bucket) / remote_key).exists()

    def delete(self, remote_key: str, bucket: str) -> None:
        """Delete a file at the given key."""
        path = Path(bucket) / remote_key
        if path.exists():
            path.unlink()
            logger.debug("delete: %s", path)

    def list_keys(self, prefix: str, bucket: str) -> list[str]:
        """List all file paths under a prefix directory."""
        base = Path(bucket) / prefix
        if not base.exists():
            return []

        keys = []
        for path in base.rglob("*"):
            if path.is_file():
                keys.append(str(path.relative_to(Path(bucket))))
        return sorted(keys)

    def read_json(self, remote_key: str, bucket: str) -> dict:
        """Read and parse a JSON file."""
        path = Path(bucket) / remote_key
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {remote_key} in {bucket}")
        return json.loads(path.read_text())

    def write_json(self, data: dict, remote_key: str, bucket: str) -> None:
        """Write a dict as JSON."""
        path = Path(bucket) / remote_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.debug("write_json: %s", path)
