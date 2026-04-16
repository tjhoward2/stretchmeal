#!/usr/bin/env python3
"""Smoke test for real PSD files.

Run against an actual rug design PSD to verify the pipeline works
end-to-end and report diagnostics for tuning detection parameters.

Usage:
    python tests/smoke_test_real_psd.py /path/to/design.psd [--design-id 001]

This script does NOT modify or archive the original file — it copies
it to a temp directory first. Safe to run on production files.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyvips

from config import PipelineConfig
from detection.border_detect import detect_borders
from ingest.flatten import flatten_psd
from ingest.validate import validate_flatten
from scaling.nine_slice import nine_slice_scale
from scaling.resize import lanczos_resize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")


def main():
    parser = argparse.ArgumentParser(description="Smoke test with a real PSD file")
    parser.add_argument("psd_path", help="Path to a real PSD file")
    parser.add_argument("--design-id", default="smoke-test")
    parser.add_argument(
        "--scale-sizes",
        nargs="*",
        default=["5x7", "8x10", "10x14"],
        help="Rug sizes to test scaling (WxH in feet), e.g. 5x7 8x10",
    )
    args = parser.parse_args()

    psd_path = Path(args.psd_path)
    if not psd_path.exists():
        logger.error("File not found: %s", psd_path)
        sys.exit(1)

    file_size_gb = psd_path.stat().st_size / (1024 ** 3)
    logger.info("Input file: %s (%.2f GB)", psd_path, file_size_gb)

    with tempfile.TemporaryDirectory(prefix="rug_smoke_") as tmpdir:
        config = PipelineConfig(base_dir=Path(tmpdir) / "data")
        config.ensure_dirs()
        design_id = args.design_id
        design_dir = config.masters_dir / design_id
        design_dir.mkdir(parents=True, exist_ok=True)

        master_tiff = design_dir / "master.tiff"
        thumbnail = design_dir / "thumbnail.png"

        # ── Step 1: Flatten ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Flattening PSD")
        logger.info("=" * 60)
        try:
            master = flatten_psd(psd_path, master_tiff, thumbnail)
        except Exception as e:
            logger.error("FLATTEN FAILED: %s", e, exc_info=True)
            sys.exit(1)

        master_size_mb = master_tiff.stat().st_size / (1024 ** 2)
        logger.info("Master TIFF: %dx%d px, %.1f MB", master.width, master.height, master_size_mb)
        logger.info("Bands: %d, Format: %s, Interpretation: %s",
                     master.bands, master.format, master.interpretation)

        # ── Step 2: Border Detection ─────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2: Border Detection")
        logger.info("=" * 60)
        zone_map = detect_borders(master, design_id)

        logger.info("Detected borders:")
        logger.info("  Top:    %d px", zone_map["border_top_px"])
        logger.info("  Bottom: %d px", zone_map["border_bottom_px"])
        logger.info("  Left:   %d px", zone_map["border_left_px"])
        logger.info("  Right:  %d px", zone_map["border_right_px"])
        logger.info("  Confidence: %.2f", zone_map["confidence"])
        logger.info("  Needs review: %s", zone_map["needs_review"])

        # Save zone map
        zone_path = design_dir / "zone_map.json"
        zone_path.write_text(json.dumps(zone_map, indent=2))

        # ── Step 3: Validation ───────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3: Validation (SSIM)")
        logger.info("=" * 60)
        validation = validate_flatten(psd_path, master_tiff)
        logger.info("SSIM score: %.4f", validation["ssim_score"])
        logger.info("Passed: %s", validation["passed"])

        # ── Step 4: Scaling ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4: 9-Slice Scaling")
        logger.info("=" * 60)

        for size_str in args.scale_sizes:
            w_ft, h_ft = map(float, size_str.split("x"))
            target_w = int(w_ft * 12 * 150)
            target_h = int(h_ft * 12 * 150)

            logger.info("Scaling to %sx%s ft (%dx%d px @ 150 DPI)...",
                        w_ft, h_ft, target_w, target_h)

            try:
                result = nine_slice_scale(
                    master, zone_map, target_w, target_h,
                    resize_fn=lanczos_resize,
                    seam_threshold=5.0,
                )
                logger.info("  Result: %dx%d px — OK", result.width, result.height)

                # Save to temp for inspection
                out_path = Path(tmpdir) / f"scaled_{size_str.replace('x', '_')}.tiff"
                result.tiffsave(str(out_path), compression="lzw")
                out_mb = out_path.stat().st_size / (1024 ** 2)
                logger.info("  Saved: %s (%.1f MB)", out_path, out_mb)
            except Exception as e:
                logger.error("  SCALING FAILED for %s: %s", size_str, e, exc_info=True)

        # ── Summary ──────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("Input:       %s (%.2f GB)", psd_path.name, file_size_gb)
        logger.info("Dimensions:  %dx%d px", master.width, master.height)
        logger.info("SSIM:        %.4f (%s)", validation["ssim_score"],
                     "PASS" if validation["passed"] else "FAIL")
        logger.info("Borders:     T=%d B=%d L=%d R=%d (conf=%.2f)",
                     zone_map["border_top_px"], zone_map["border_bottom_px"],
                     zone_map["border_left_px"], zone_map["border_right_px"],
                     zone_map["confidence"])
        logger.info("Review:      %s", "YES" if zone_map["needs_review"] else "No")
        logger.info("")
        logger.info("Thumbnail saved to: %s", thumbnail)
        logger.info("(temp directory will be cleaned up on exit)")


if __name__ == "__main__":
    main()
