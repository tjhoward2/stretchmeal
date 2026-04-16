"""Tests for the 9-slice scaling engine."""

import logging

import numpy as np
import pyvips
import pytest

from scaling.nine_slice import default_resize, nine_slice_scale


def _make_test_image(width: int, height: int, bands: int = 3) -> pyvips.Image:
    """Create a synthetic test image with distinct border and interior colors.

    Border region (outer 50px): red (255, 0, 0)
    Interior: blue (0, 0, 255)
    """
    border = 50
    arr = np.zeros((height, width, bands), dtype=np.uint8)

    # Fill interior blue
    arr[border:height - border, border:width - border] = [0, 0, 255][:bands]

    # Fill borders red
    arr[:border, :] = [255, 0, 0][:bands]           # top
    arr[height - border:, :] = [255, 0, 0][:bands]  # bottom
    arr[:, :border] = [255, 0, 0][:bands]            # left
    arr[:, width - border:] = [255, 0, 0][:bands]    # right

    return pyvips.Image.new_from_memory(
        arr.tobytes(), width, height, bands, "uchar"
    )


def _zone_map(
    border: int = 50, width: int = 400, height: int = 600
) -> dict:
    return {
        "border_top_px": border,
        "border_bottom_px": border,
        "border_left_px": border,
        "border_right_px": border,
        "master_width_px": width,
        "master_height_px": height,
    }


class TestNineSliceScale:
    """Test suite for nine_slice_scale."""

    def test_output_dimensions_match_target(self):
        """Result image should exactly match requested dimensions."""
        img = _make_test_image(400, 600)
        result = nine_slice_scale(img, _zone_map(), 300, 400)
        assert result.width == 300
        assert result.height == 400

    def test_output_dimensions_upscale(self):
        """Upscaling should also produce exact target dimensions."""
        img = _make_test_image(400, 600)
        result = nine_slice_scale(img, _zone_map(), 800, 1200)
        assert result.width == 800
        assert result.height == 1200

    def test_corners_preserved(self):
        """Corner pixels should be identical to the source."""
        img = _make_test_image(400, 600)
        zone = _zone_map()
        result = nine_slice_scale(img, zone, 600, 900)

        # Top-left corner: should still be red
        tl = np.ndarray(
            buffer=result.crop(0, 0, 1, 1).write_to_memory(),
            dtype=np.uint8,
            shape=(1, 1, 3),
        )
        assert tl[0, 0, 0] == 255  # red channel
        assert tl[0, 0, 1] == 0
        assert tl[0, 0, 2] == 0

    def test_interior_color_preserved(self):
        """Interior center pixel should remain blue after scaling."""
        img = _make_test_image(400, 600)
        zone = _zone_map()
        result = nine_slice_scale(img, zone, 600, 900)

        cx = result.width // 2
        cy = result.height // 2
        center = np.ndarray(
            buffer=result.crop(cx, cy, 1, 1).write_to_memory(),
            dtype=np.uint8,
            shape=(1, 1, 3),
        )
        assert center[0, 0, 0] == 0
        assert center[0, 0, 1] == 0
        assert center[0, 0, 2] == 255

    def test_zero_borders(self):
        """Borderless design: entire image is the interior."""
        img = _make_test_image(400, 600)
        zone = _zone_map(border=0)
        result = nine_slice_scale(img, zone, 200, 300)
        assert result.width == 200
        assert result.height == 300

    def test_fallback_when_target_too_small(self, caplog):
        """When target interior < 100px, should fall back to uniform scaling."""
        img = _make_test_image(400, 600)
        zone = _zone_map(border=100)  # 200px of borders, leaves very little
        with caplog.at_level(logging.WARNING):
            result = nine_slice_scale(img, zone, 250, 250)
        assert result.width == 250
        assert result.height == 250
        assert "Falling back to uniform scaling" in caplog.text

    def test_custom_resize_fn_called(self):
        """Custom resize_fn should be used for all resize operations."""
        calls = []

        def tracking_resize(image, tw, th):
            calls.append((tw, th))
            return default_resize(image, tw, th)

        img = _make_test_image(400, 600)
        zone = _zone_map()
        nine_slice_scale(img, zone, 600, 900, resize_fn=tracking_resize)
        # Should have calls for: top strip, bottom strip, left strip,
        # right strip, and interior = 5 resize calls
        assert len(calls) == 5

    def test_seam_warning_logged(self, caplog):
        """Seam check should warn when there's a color discontinuity."""
        # Create an image with an intentional seam problem:
        # border is white, interior is black — big jump at boundary
        w, h, border = 400, 600, 50
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:border, :] = 255
        arr[h - border:, :] = 255
        arr[:, :border] = 255
        arr[:, w - border:] = 255

        img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
        zone = _zone_map()

        with caplog.at_level(logging.WARNING):
            nine_slice_scale(img, zone, 600, 900, seam_threshold=1.0)

        assert "avg color diff" in caplog.text

    def test_same_size_passthrough(self):
        """Scaling to the same size should produce valid output."""
        img = _make_test_image(400, 600)
        result = nine_slice_scale(img, _zone_map(), 400, 600)
        assert result.width == 400
        assert result.height == 600

    def test_asymmetric_borders(self):
        """Different border sizes on each side should work correctly."""
        img = _make_test_image(400, 600)
        zone = {
            "border_top_px": 30,
            "border_bottom_px": 60,
            "border_left_px": 40,
            "border_right_px": 20,
            "master_width_px": 400,
            "master_height_px": 600,
        }
        result = nine_slice_scale(img, zone, 500, 700)
        assert result.width == 500
        assert result.height == 700


class TestDefaultResize:
    """Test the default Lanczos resize function."""

    def test_downscale(self):
        img = _make_test_image(400, 600)
        result = default_resize(img, 200, 300)
        assert result.width == 200
        assert result.height == 300

    def test_identity(self):
        img = _make_test_image(400, 600)
        result = default_resize(img, 400, 600)
        assert result.width == 400
        assert result.height == 600
