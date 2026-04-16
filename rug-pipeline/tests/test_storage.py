"""Tests for storage backends (local and S3)."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from storage.backend import create_backend
from storage.local import LocalBackend


class TestLocalBackend:
    """Test the local filesystem storage backend."""

    def test_put_and_get(self, tmp_path):
        """Should copy a file to the bucket path and retrieve it."""
        backend = LocalBackend()
        bucket = str(tmp_path / "bucket")

        # Create a source file
        source = tmp_path / "source.txt"
        source.write_text("hello world")

        # Put
        backend.put(source, "designs/001/data.txt", bucket)
        assert (Path(bucket) / "designs/001/data.txt").exists()
        assert (Path(bucket) / "designs/001/data.txt").read_text() == "hello world"

        # Get
        dest = tmp_path / "downloaded.txt"
        backend.get("designs/001/data.txt", bucket, dest)
        assert dest.read_text() == "hello world"

    def test_get_missing_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing keys."""
        backend = LocalBackend()
        with pytest.raises(FileNotFoundError):
            backend.get("nonexistent", str(tmp_path), tmp_path / "out.txt")

    def test_exists(self, tmp_path):
        """Should report existence correctly."""
        backend = LocalBackend()
        bucket = str(tmp_path / "bucket")

        assert backend.exists("anything", bucket) is False

        source = tmp_path / "f.txt"
        source.write_text("x")
        backend.put(source, "f.txt", bucket)
        assert backend.exists("f.txt", bucket) is True

    def test_delete(self, tmp_path):
        """Should delete files."""
        backend = LocalBackend()
        bucket = str(tmp_path / "bucket")

        source = tmp_path / "f.txt"
        source.write_text("x")
        backend.put(source, "f.txt", bucket)
        assert backend.exists("f.txt", bucket)

        backend.delete("f.txt", bucket)
        assert not backend.exists("f.txt", bucket)

    def test_list_keys(self, tmp_path):
        """Should list all files under a prefix."""
        backend = LocalBackend()
        bucket = str(tmp_path / "bucket")

        source = tmp_path / "f.txt"
        source.write_text("x")
        backend.put(source, "designs/001/master.tiff", bucket)
        backend.put(source, "designs/001/zone_map.json", bucket)
        backend.put(source, "designs/002/master.tiff", bucket)

        keys = backend.list_keys("designs/001", bucket)
        assert len(keys) == 2
        assert "designs/001/master.tiff" in keys
        assert "designs/001/zone_map.json" in keys

    def test_read_write_json(self, tmp_path):
        """Should write and read JSON correctly."""
        backend = LocalBackend()
        bucket = str(tmp_path / "bucket")

        data = {"design_id": "001", "confidence": 0.95, "needs_review": False}
        backend.write_json(data, "001/zone_map.json", bucket)

        result = backend.read_json("001/zone_map.json", bucket)
        assert result == data

    def test_put_same_path_is_noop(self, tmp_path):
        """Putting a file to its own location should not error."""
        backend = LocalBackend()
        bucket = str(tmp_path)

        f = tmp_path / "test.txt"
        f.write_text("content")

        # Put the file to its own path
        backend.put(f, "test.txt", str(tmp_path))
        assert f.read_text() == "content"


class TestS3Backend:
    """Test S3 backend using moto mock.

    These tests require moto[s3] to be installed. They are skipped
    if moto is not available.
    """

    @pytest.fixture
    def s3_backend(self):
        """Create an S3Backend with mocked AWS."""
        try:
            from moto import mock_aws
        except ImportError:
            pytest.skip("moto not installed — skipping S3 tests")

        with mock_aws():
            os.environ["AWS_ACCESS_KEY_ID"] = "testing"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
            os.environ["AWS_SECURITY_TOKEN"] = "testing"
            os.environ["AWS_SESSION_TOKEN"] = "testing"
            os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

            from storage.s3 import S3Backend
            backend = S3Backend(region="us-east-1", glacier_transition_days=1)
            yield backend

    def test_put_and_get(self, s3_backend, tmp_path):
        """Should upload to S3 and download back."""
        source = tmp_path / "test.txt"
        source.write_text("s3 content")

        s3_backend.put(source, "designs/001/test.txt", "test-bucket")

        dest = tmp_path / "downloaded.txt"
        s3_backend.get("designs/001/test.txt", "test-bucket", dest)
        assert dest.read_text() == "s3 content"

    def test_exists(self, s3_backend, tmp_path):
        """Should report S3 object existence."""
        assert s3_backend.exists("nope", "test-bucket") is False

        source = tmp_path / "f.txt"
        source.write_text("x")
        s3_backend.put(source, "f.txt", "test-bucket")
        assert s3_backend.exists("f.txt", "test-bucket") is True

    def test_delete(self, s3_backend, tmp_path):
        """Should delete S3 objects."""
        source = tmp_path / "f.txt"
        source.write_text("x")
        s3_backend.put(source, "f.txt", "test-bucket")
        assert s3_backend.exists("f.txt", "test-bucket")

        s3_backend.delete("f.txt", "test-bucket")
        assert not s3_backend.exists("f.txt", "test-bucket")

    def test_list_keys(self, s3_backend, tmp_path):
        """Should list S3 objects under a prefix."""
        source = tmp_path / "f.txt"
        source.write_text("x")
        s3_backend.put(source, "designs/001/master.tiff", "test-bucket")
        s3_backend.put(source, "designs/001/zone.json", "test-bucket")
        s3_backend.put(source, "designs/002/master.tiff", "test-bucket")

        keys = s3_backend.list_keys("designs/001/", "test-bucket")
        assert len(keys) == 2

    def test_read_write_json(self, s3_backend):
        """Should write and read JSON via S3."""
        data = {"id": "001", "value": 42}
        s3_backend.write_json(data, "test.json", "test-bucket")
        result = s3_backend.read_json("test.json", "test-bucket")
        assert result == data

    def test_glacier_lifecycle(self, s3_backend, tmp_path):
        """Should configure Glacier lifecycle rule on a bucket."""
        # Create bucket first
        source = tmp_path / "f.txt"
        source.write_text("x")
        s3_backend.put(source, "test.txt", "archive-bucket")

        s3_backend.configure_glacier_lifecycle("archive-bucket")

        # Verify lifecycle was set
        response = s3_backend._s3.get_bucket_lifecycle_configuration(
            Bucket="archive-bucket"
        )
        rules = response["Rules"]
        assert any(r["ID"] == "rug-pipeline-glacier-archive" for r in rules)

        # Calling again should be idempotent
        s3_backend.configure_glacier_lifecycle("archive-bucket")

    def test_get_missing_raises(self, s3_backend, tmp_path):
        """Should raise FileNotFoundError for missing S3 keys."""
        # Create bucket first
        source = tmp_path / "f.txt"
        source.write_text("x")
        s3_backend.put(source, "exists.txt", "test-bucket")

        with pytest.raises(FileNotFoundError):
            s3_backend.get("missing.txt", "test-bucket", tmp_path / "out.txt")


class TestCreateBackend:
    """Test the backend factory."""

    def test_create_local(self):
        backend = create_backend("local")
        assert isinstance(backend, LocalBackend)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_backend("gcs")
