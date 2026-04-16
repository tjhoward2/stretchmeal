"""Integration test: full pipeline from ingest through scaling.

Creates a synthetic rug image with a clear border, ingests it,
detects borders, scales to 3 sizes, and verifies outputs.
"""

import json
from pathlib import Path

import numpy as np
import pyvips
import pytest

from config import PipelineConfig
from detection.border_detect import detect_borders
from ingest.flatten import flatten_psd
from ingest.validate import validate_flatten
from scaling.cache import check_cache, write_cache
from scaling.nine_slice import nine_slice_scale
from scaling.resize import lanczos_resize


def _create_synthetic_rug(path: Path, width: int = 1800, height: int = 2520,
                          border: int = 200) -> None:
    """Create a synthetic rug image with a clearly defined border.

    - Corners: gold pattern (200, 170, 50)
    - Border strips: dark red with horizontal/vertical lines (150, 40, 40)
    - Interior: rich blue with texture (40, 60, 160)
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Interior — blue with a subtle gradient for realism
    for y in range(border, height - border):
        for x in range(border, width - border):
            arr[y, x] = [
                40 + int(20 * np.sin(x / 50)),
                60 + int(20 * np.sin(y / 50)),
                160,
            ]

    # Border strips — dark red with periodic lines
    # Top/bottom
    for y in list(range(0, border)) + list(range(height - border, height)):
        for x in range(width):
            base = [150, 40, 40]
            if y % 20 < 2:  # horizontal lines
                base = [180, 60, 60]
            arr[y, x] = base

    # Left/right
    for x in list(range(0, border)) + list(range(width - border, width)):
        for y in range(border, height - border):
            base = [150, 40, 40]
            if x % 20 < 2:  # vertical lines
                base = [180, 60, 60]
            arr[y, x] = base

    # Corners — gold
    for y in range(border):
        for x in range(border):
            arr[y, x] = [200, 170, 50]
        for x in range(width - border, width):
            arr[y, x] = [200, 170, 50]
    for y in range(height - border, height):
        for x in range(border):
            arr[y, x] = [200, 170, 50]
        for x in range(width - border, width):
            arr[y, x] = [200, 170, 50]

    img = pyvips.Image.new_from_memory(arr.tobytes(), width, height, 3, "uchar")
    img.tiffsave(str(path), compression="lzw")


class TestFullPipeline:
    """End-to-end integration test."""

    @pytest.fixture
    def pipeline_env(self, tmp_path):
        """Set up a complete pipeline environment with a synthetic rug."""
        config = PipelineConfig(base_dir=tmp_path / "pipeline_data")
        config.ensure_dirs()

        # Create synthetic rug as a TIFF (stand-in for PSD)
        source = tmp_path / "rug_design.tiff"
        _create_synthetic_rug(source, width=1800, height=2520, border=200)

        return config, source, "test-001"

    def test_ingest_creates_all_outputs(self, pipeline_env):
        """Ingest should create master TIFF, thumbnail, and zone map."""
        config, source, design_id = pipeline_env
        design_dir = config.masters_dir / design_id

        # Flatten
        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"
        master = flatten_psd(source, master_tiff, thumbnail, config.thumbnail_width_px)

        assert master_tiff.exists()
        assert thumbnail.exists()
        assert master.width == 1800
        assert master.height == 2520

        # Thumbnail is ~1000px wide
        thumb_img = pyvips.Image.new_from_file(str(thumbnail))
        assert 990 <= thumb_img.width <= 1010

    def test_border_detection_accuracy(self, pipeline_env):
        """Border detection on synthetic rug should be close to ground truth."""
        config, source, design_id = pipeline_env
        design_dir = config.masters_dir / design_id

        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"
        master = flatten_psd(source, master_tiff, thumbnail)

        zone_map = detect_borders(master, design_id,
                                  detection_width=config.border_detection_width_px)

        # Ground truth border is 200px on all sides
        for key in ("border_top_px", "border_bottom_px",
                    "border_left_px", "border_right_px"):
            detected = zone_map[key]
            assert 120 <= detected <= 280, (
                f"{key}={detected}, expected near 200"
            )

        assert zone_map["confidence"] > 0.5

    def test_validation_passes(self, pipeline_env):
        """Validation should pass when comparing TIFF to itself."""
        config, source, design_id = pipeline_env
        design_dir = config.masters_dir / design_id

        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"
        flatten_psd(source, master_tiff, thumbnail)

        result = validate_flatten(source, master_tiff,
                                  ssim_threshold=config.validation_ssim_threshold)
        assert result["passed"] is True

    def test_scale_to_multiple_sizes(self, pipeline_env):
        """Scaling to 3 different sizes should produce correct dimensions."""
        config, source, design_id = pipeline_env
        design_dir = config.masters_dir / design_id

        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"
        master = flatten_psd(source, master_tiff, thumbnail)

        zone_map = detect_borders(master, design_id,
                                  detection_width=config.border_detection_width_px)

        # Save zone map (as the CLI would)
        zone_map_path = design_dir / "zone_map.json"
        zone_map_path.write_text(json.dumps(zone_map, indent=2))

        # Scale to 3 sizes
        test_sizes = [
            (5, 7, 150),   # 9000 x 12600 px — smaller than master
            (8, 10, 150),  # 14400 x 18000 px — close to master
            (3, 5, 100),   # 3600 x 6000 px — much smaller, lower DPI
        ]

        for width_ft, height_ft, dpi in test_sizes:
            target_w = int(width_ft * 12 * dpi)
            target_h = int(height_ft * 12 * dpi)

            result = nine_slice_scale(
                master, zone_map, target_w, target_h,
                resize_fn=lanczos_resize,
                seam_threshold=config.seam_threshold,
            )

            assert result.width == target_w, (
                f"Size {width_ft}x{height_ft}@{dpi}: "
                f"width {result.width} != {target_w}"
            )
            assert result.height == target_h, (
                f"Size {width_ft}x{height_ft}@{dpi}: "
                f"height {result.height} != {target_h}"
            )

            # Cache the output
            cached_path = write_cache(
                result, config.cache_dir, design_id,
                width_ft, height_ft, dpi,
            )
            assert cached_path.exists()

    def test_cache_hit_after_scaling(self, pipeline_env):
        """After scaling, the cache should return a hit for the same size."""
        config, source, design_id = pipeline_env
        design_dir = config.masters_dir / design_id

        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"
        master = flatten_psd(source, master_tiff, thumbnail)

        zone_map = detect_borders(master, design_id)

        target_w = int(5 * 12 * 150)
        target_h = int(7 * 12 * 150)
        result = nine_slice_scale(master, zone_map, target_w, target_h,
                                  resize_fn=lanczos_resize)

        write_cache(result, config.cache_dir, design_id, 5.0, 7.0, 150)

        # Cache should now return a hit
        hit = check_cache(config.cache_dir, design_id, 5.0, 7.0, 150)
        assert hit is not None
        assert hit.exists()

    def test_borderless_design_scales_correctly(self, tmp_path):
        """A design with no border should scale as one piece."""
        config = PipelineConfig(base_dir=tmp_path / "data")
        config.ensure_dirs()

        # Uniform image — no border
        arr = np.full((600, 400, 3), 100, dtype=np.uint8)
        img = pyvips.Image.new_from_memory(arr.tobytes(), 400, 600, 3, "uchar")

        zone_map = {
            "border_top_px": 0,
            "border_bottom_px": 0,
            "border_left_px": 0,
            "border_right_px": 0,
            "master_width_px": 400,
            "master_height_px": 600,
        }

        result = nine_slice_scale(img, zone_map, 200, 300,
                                  resize_fn=lanczos_resize)
        assert result.width == 200
        assert result.height == 300
