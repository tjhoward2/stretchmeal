"""Tests for resize orchestration and cache management."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvips
import pytest

from scaling.resize import lanczos_resize, make_esrgan_resize, needs_upscale
from scaling.cache import (
    cache_path,
    check_cache,
    cleanup_cache,
    metadata_path,
    write_cache,
)


def _make_image(w: int = 200, h: int = 300) -> pyvips.Image:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")


class TestLanczosResize:

    def test_downscale(self):
        img = _make_image(400, 600)
        result = lanczos_resize(img, 200, 300)
        assert result.width == 200
        assert result.height == 300

    def test_upscale(self):
        img = _make_image(200, 300)
        result = lanczos_resize(img, 400, 600)
        assert result.width == 400
        assert result.height == 600

    def test_identity(self):
        img = _make_image(200, 300)
        result = lanczos_resize(img, 200, 300)
        assert result is img  # should return same object


class TestEsrganResize:

    def test_fallback_when_binary_missing(self):
        """Should return lanczos_resize when binary not found."""
        with patch("shutil.which", return_value=None):
            fn = make_esrgan_resize(binary="nonexistent")
        # The returned function should work as Lanczos fallback
        img = _make_image()
        result = fn(img, 100, 150)
        assert result.width == 100
        assert result.height == 150


class TestNeedsUpscale:

    def test_upscale_needed(self):
        assert needs_upscale(1000, 1000, 2000, 1000) is True
        assert needs_upscale(1000, 1000, 1000, 2000) is True

    def test_no_upscale(self):
        assert needs_upscale(1000, 1000, 500, 500) is False
        assert needs_upscale(1000, 1000, 1000, 1000) is False


class TestCachePath:

    def test_path_format(self):
        p = cache_path(Path("/cache"), "142", 5.0, 7.0, 150)
        assert p == Path("/cache/142/5.0x7.0_150dpi.tiff")

    def test_metadata_path(self):
        p = cache_path(Path("/cache"), "142", 5.0, 7.0, 150)
        assert metadata_path(p) == Path("/cache/142/5.0x7.0_150dpi.json")


class TestCheckCache:

    def test_returns_none_when_missing(self, tmp_path):
        result = check_cache(tmp_path, "999", 5.0, 7.0, 150)
        assert result is None

    def test_returns_path_when_valid(self, tmp_path):
        # Create a cached file
        img = _make_image()
        path = write_cache(img, tmp_path, "142", 5.0, 7.0, 150)

        result = check_cache(tmp_path, "142", 5.0, 7.0, 150)
        assert result == path

    def test_returns_none_when_stale(self, tmp_path):
        img = _make_image()
        path = write_cache(img, tmp_path, "142", 5.0, 7.0, 150)

        # Make the file appear old
        old_time = time.time() - (31 * 86400)
        os.utime(path, (old_time, old_time))

        result = check_cache(tmp_path, "142", 5.0, 7.0, 150, max_age_days=30)
        assert result is None


class TestWriteCache:

    def test_creates_tiff_and_metadata(self, tmp_path):
        img = _make_image(400, 600)
        path = write_cache(img, tmp_path, "142", 5.0, 7.0, 150, needs_approval=True)

        assert path.exists()
        meta_path_val = metadata_path(path)
        assert meta_path_val.exists()

        meta = json.loads(meta_path_val.read_text())
        assert meta["design_id"] == "142"
        assert meta["needs_approval"] is True
        assert meta["width_px"] == 400
        assert meta["height_px"] == 600


class TestCleanupCache:

    def test_deletes_old_files(self, tmp_path):
        img = _make_image()
        path = write_cache(img, tmp_path, "old", 5.0, 7.0, 150)

        # Make old
        old_time = time.time() - (31 * 86400)
        os.utime(path, (old_time, old_time))
        os.utime(metadata_path(path), (old_time, old_time))

        deleted = cleanup_cache(tmp_path, max_age_days=30)
        assert len(deleted) >= 1
        assert not path.exists()

    def test_keeps_fresh_files(self, tmp_path):
        img = _make_image()
        path = write_cache(img, tmp_path, "fresh", 5.0, 7.0, 150)

        deleted = cleanup_cache(tmp_path, max_age_days=30)
        assert len(deleted) == 0
        assert path.exists()

    def test_empty_cache_dir(self, tmp_path):
        deleted = cleanup_cache(tmp_path / "nonexistent", max_age_days=30)
        assert deleted == []
