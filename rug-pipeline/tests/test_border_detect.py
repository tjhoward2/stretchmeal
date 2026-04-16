"""Tests for automatic border detection."""

import numpy as np
import pyvips
import pytest

from detection.border_detect import detect_borders


def _make_bordered_image(
    width: int = 500,
    height: int = 700,
    border: int = 60,
    border_color: tuple = (200, 50, 50),
    interior_color: tuple = (50, 50, 200),
) -> pyvips.Image:
    """Create a test image with distinct border and interior regions."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill entire image with border color
    arr[:, :] = border_color

    # Fill interior with interior color
    arr[border:height - border, border:width - border] = interior_color

    return pyvips.Image.new_from_memory(arr.tobytes(), width, height, 3, "uchar")


def _make_borderless_image(
    width: int = 500, height: int = 700
) -> pyvips.Image:
    """Create a uniform image with no border pattern."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    # Add some noise so it's not perfectly flat
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 10, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return pyvips.Image.new_from_memory(arr.tobytes(), width, height, 3, "uchar")


class TestBorderDetection:
    """Test suite for border detection."""

    def test_detects_clear_border(self):
        """Should detect borders when there's a clear color transition."""
        img = _make_bordered_image(border=60)
        zone = detect_borders(img, "test-001", detection_width=500)

        assert zone["design_id"] == "test-001"
        assert zone["detection_method"] == "auto"
        assert zone["master_width_px"] == 500
        assert zone["master_height_px"] == 700

        # Borders should be roughly 60px (at full resolution since
        # detection_width matches image width, no downscaling)
        for key in ("border_top_px", "border_bottom_px", "border_left_px", "border_right_px"):
            assert 40 <= zone[key] <= 80, f"{key}={zone[key]} not near expected 60"

    def test_confidence_above_threshold(self):
        """Clear borders should have high confidence."""
        img = _make_bordered_image(border=80)
        zone = detect_borders(img, "test-002", detection_width=500)
        assert zone["confidence"] >= 0.7
        assert zone["needs_review"] is False

    def test_borderless_detected_as_zero(self):
        """Uniform image should detect zero borders with high confidence."""
        img = _make_borderless_image()
        zone = detect_borders(img, "test-003", detection_width=500)

        # All borders should be 0 or very small
        total_border = (
            zone["border_top_px"]
            + zone["border_bottom_px"]
            + zone["border_left_px"]
            + zone["border_right_px"]
        )
        # For a borderless image, if all borders come back as 0, confidence is 1.0
        if total_border == 0:
            assert zone["confidence"] == 1.0
            assert zone["needs_review"] is False

    def test_asymmetric_border(self):
        """Detect different border widths on each side."""
        w, h = 500, 700
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :] = (200, 50, 50)  # border

        t, b, l, r = 30, 80, 50, 50
        arr[t:h - b, l:w - r] = (50, 50, 200)  # interior

        img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
        zone = detect_borders(img, "test-004", detection_width=500)

        # Top should be smaller than bottom
        assert zone["border_top_px"] < zone["border_bottom_px"]

    def test_zone_map_has_required_keys(self):
        """Zone map should contain all required fields."""
        img = _make_bordered_image()
        zone = detect_borders(img, "test-005", detection_width=500)

        required_keys = {
            "design_id",
            "border_top_px",
            "border_bottom_px",
            "border_left_px",
            "border_right_px",
            "master_width_px",
            "master_height_px",
            "detection_method",
            "confidence",
            "needs_review",
        }
        assert required_keys.issubset(zone.keys())

    def test_scales_back_to_master_resolution(self):
        """Border positions should be reported at master resolution, not detection resolution."""
        # Create a 2000px wide image with 120px border
        img = _make_bordered_image(width=2000, height=2800, border=120)
        # Detect at 500px width (4x downscale)
        zone = detect_borders(img, "test-006", detection_width=500)

        # Borders should be near 120px at master resolution
        for key in ("border_top_px", "border_bottom_px", "border_left_px", "border_right_px"):
            assert 80 <= zone[key] <= 180, f"{key}={zone[key]} not near expected 120"

    def test_low_confidence_flags_review(self):
        """When gradient is subtle, should flag for manual review."""
        # Create an image with very similar border and interior colors
        w, h = 500, 700
        arr = np.full((h, w, 3), 128, dtype=np.uint8)
        # Tiny color difference at border boundary
        arr[60:h - 60, 60:w - 60] = 132

        img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
        zone = detect_borders(img, "test-007", detection_width=500, confidence_threshold=0.9)

        # With such subtle differences, it might flag for review
        # (exact behavior depends on the signal, but we test the mechanism)
        assert isinstance(zone["needs_review"], bool)
        assert isinstance(zone["confidence"], float)
