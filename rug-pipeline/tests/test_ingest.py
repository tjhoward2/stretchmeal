"""Tests for PSD flattening and validation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyvips
import pytest

from ingest.flatten import flatten_psd
from ingest.validate import validate_flatten


def _create_test_tiff(path: Path, width: int = 400, height: int = 600) -> None:
    """Create a synthetic TIFF file to act as a 'PSD' for testing.

    pyvips can read TIFF files the same way, so we use TIFF as a stand-in
    since creating real PSD files requires specialized tools.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a gradient pattern so SSIM has something to compare
    for y in range(height):
        for x in range(width):
            arr[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128,
            ]
    img = pyvips.Image.new_from_memory(arr.tobytes(), width, height, 3, "uchar")
    img.tiffsave(str(path), compression="lzw")


class TestFlatten:
    """Test PSD flattening (using TIFF as PSD stand-in)."""

    def test_flatten_creates_tiff_and_thumbnail(self, tmp_path):
        """Should create both master TIFF and thumbnail PNG."""
        source = tmp_path / "input.tiff"
        _create_test_tiff(source)

        output_tiff = tmp_path / "masters" / "master.tiff"
        thumb = tmp_path / "masters" / "thumbnail.png"

        result = flatten_psd(source, output_tiff, thumb, thumbnail_width=200)

        assert output_tiff.exists()
        assert thumb.exists()
        assert result.width == 400
        assert result.height == 600

    def test_thumbnail_is_correct_width(self, tmp_path):
        """Thumbnail should be the requested width."""
        source = tmp_path / "input.tiff"
        _create_test_tiff(source, width=800, height=1200)

        output_tiff = tmp_path / "master.tiff"
        thumb = tmp_path / "thumb.png"

        flatten_psd(source, output_tiff, thumb, thumbnail_width=200)

        thumb_img = pyvips.Image.new_from_file(str(thumb))
        assert thumb_img.width == 200

    def test_flatten_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            flatten_psd(
                tmp_path / "nonexistent.psd",
                tmp_path / "out.tiff",
                tmp_path / "thumb.png",
            )


class TestValidation:
    """Test SSIM validation."""

    def test_identical_images_pass(self, tmp_path):
        """Same image compared to itself should score ~1.0."""
        img_path = tmp_path / "test.tiff"
        _create_test_tiff(img_path)

        result = validate_flatten(img_path, img_path)
        assert result["passed"] is True
        assert result["ssim_score"] >= 0.99

    def test_different_images_fail(self, tmp_path):
        """Clearly different images should fail validation."""
        img1 = tmp_path / "img1.tiff"
        img2 = tmp_path / "img2.tiff"

        # Create two very different images
        arr1 = np.zeros((600, 400, 3), dtype=np.uint8)
        arr1[:, :] = (255, 0, 0)
        pyvips.Image.new_from_memory(
            arr1.tobytes(), 400, 600, 3, "uchar"
        ).tiffsave(str(img1))

        arr2 = np.zeros((600, 400, 3), dtype=np.uint8)
        arr2[:, :] = (0, 0, 255)
        pyvips.Image.new_from_memory(
            arr2.tobytes(), 400, 600, 3, "uchar"
        ).tiffsave(str(img2))

        result = validate_flatten(img1, img2, ssim_threshold=0.95)
        assert result["passed"] is False
        assert result["needs_review"] is True

    def test_result_has_required_fields(self, tmp_path):
        """Validation result should have all expected fields."""
        img_path = tmp_path / "test.tiff"
        _create_test_tiff(img_path)

        result = validate_flatten(img_path, img_path)
        assert "ssim_score" in result
        assert "passed" in result
        assert "needs_review" in result
        assert "threshold" in result
