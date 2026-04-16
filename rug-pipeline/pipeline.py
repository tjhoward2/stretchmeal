#!/usr/bin/env python3
"""CLI entry point for the rug design pipeline.

Commands:
    ingest          Flatten a PSD, detect borders, validate, and archive.
    scale           Generate a print-ready file for a specific rug size.
    set-zones       Manually set border zones for a design.
    cleanup-cache   Delete cached files older than N days.
    list            List all designs and their status.
    validate        Validate a design's master against its original.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path so modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from detection.border_detect import detect_borders
from ingest.flatten import flatten_psd
from ingest.validate import validate_flatten
from scaling.cache import check_cache, cleanup_cache, write_cache
from scaling.nine_slice import nine_slice_scale
from scaling.resize import lanczos_resize, make_esrgan_resize, needs_upscale

logger = logging.getLogger("pipeline")


def cmd_ingest(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Ingest a PSD: flatten, detect borders, validate, archive."""
    psd_path = Path(args.psd)
    design_id = args.design_id

    if not psd_path.exists():
        logger.error("PSD file not found: %s", psd_path)
        sys.exit(1)

    config.ensure_dirs()
    design_dir = config.masters_dir / design_id
    design_dir.mkdir(parents=True, exist_ok=True)

    master_tiff = design_dir / "master.tiff"
    thumbnail = design_dir / "thumbnail.png"
    zone_map_path = design_dir / "zone_map.json"

    # Step 1: Flatten PSD
    logger.info("Step 1/4: Flattening PSD...")
    try:
        master = flatten_psd(psd_path, master_tiff, thumbnail, config.thumbnail_width_px)
    except Exception as e:
        logger.error("Flattening failed: %s. Original PSD left in place.", e)
        sys.exit(1)

    # Step 2: Border detection
    logger.info("Step 2/4: Detecting borders...")
    zone_map = detect_borders(
        master,
        design_id,
        detection_width=config.border_detection_width_px,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
        gradient_weight=config.gradient_weight,
        edge_weight=config.edge_weight,
        confidence_threshold=config.border_confidence_threshold,
    )

    if zone_map["confidence"] < config.border_confidence_threshold:
        logger.warning(
            "Border detection confidence %.2f is below threshold %.2f. "
            "Design flagged for manual zone tagging.",
            zone_map["confidence"],
            config.border_confidence_threshold,
        )

    zone_map_path.write_text(json.dumps(zone_map, indent=2))
    logger.info("Zone map saved: %s", zone_map_path)

    # Step 3: Validation
    logger.info("Step 3/4: Validating flattened output...")
    validation = validate_flatten(
        psd_path, master_tiff,
        ssim_threshold=config.validation_ssim_threshold,
    )

    if not validation["passed"]:
        logger.warning(
            "Validation SSIM %.4f below threshold %.2f. Flagging for review.",
            validation["ssim_score"],
            config.validation_ssim_threshold,
        )
        zone_map["needs_review"] = True
        zone_map_path.write_text(json.dumps(zone_map, indent=2))

    # Step 4: Archive original PSD
    logger.info("Step 4/4: Archiving original PSD...")
    archive_dir = config.archive_dir / design_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_dest = archive_dir / f"original{psd_path.suffix}"
    shutil.move(str(psd_path), str(archive_dest))
    logger.info("Archived PSD to: %s", archive_dest)

    logger.info(
        "Ingest complete for design %s. Master: %s",
        design_id, master_tiff,
    )


def cmd_scale(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Generate a print-ready file at a specific rug size."""
    design_id = args.design_id
    width_ft = args.width
    height_ft = args.height
    dpi = args.dpi

    config.ensure_dirs()

    # Check cache first
    cached = check_cache(config.cache_dir, design_id, width_ft, height_ft, dpi, config.cache_max_age_days)
    if cached:
        logger.info("Using cached file: %s", cached)
        print(str(cached))
        return

    # Load master and zone map
    design_dir = config.masters_dir / design_id
    master_path = design_dir / "master.tiff"
    zone_map_path = design_dir / "zone_map.json"

    if not master_path.exists():
        logger.error("Master TIFF not found for design %s", design_id)
        sys.exit(1)

    if not zone_map_path.exists():
        logger.error("Zone map not found for design %s", design_id)
        sys.exit(1)

    master = __import__("pyvips").Image.new_from_file(str(master_path))
    zone_map = json.loads(zone_map_path.read_text())

    # Calculate target pixels
    target_w = int(width_ft * 12 * dpi)
    target_h = int(height_ft * 12 * dpi)

    logger.info(
        "Scaling design %s to %sx%s ft (%dx%d px at %d DPI)",
        design_id, width_ft, height_ft, target_w, target_h, dpi,
    )

    # Choose resize function
    upscale = needs_upscale(master.width, master.height, target_w, target_h)
    if upscale:
        logger.info("Upscale needed — attempting Real-ESRGAN")
        resize_fn = make_esrgan_resize(
            config.realesrgan_binary, config.realesrgan_scale, config.realesrgan_model,
        )
    else:
        resize_fn = lanczos_resize

    # 9-slice scale
    result = nine_slice_scale(
        master, zone_map, target_w, target_h,
        resize_fn=resize_fn,
        seam_threshold=config.seam_threshold,
    )

    # Save to cache
    output_path = write_cache(
        result, config.cache_dir, design_id,
        width_ft, height_ft, dpi,
        needs_approval=upscale,
    )

    logger.info("Output saved: %s", output_path)
    print(str(output_path))


def cmd_set_zones(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Manually set border zones for a design."""
    design_id = args.design_id
    design_dir = config.masters_dir / design_id
    zone_map_path = design_dir / "zone_map.json"
    master_path = design_dir / "master.tiff"

    if not master_path.exists():
        logger.error("Master TIFF not found for design %s", design_id)
        sys.exit(1)

    # Read master dimensions
    import pyvips
    master = pyvips.Image.new_from_file(str(master_path))

    zone_map = {
        "design_id": design_id,
        "border_top_px": args.top,
        "border_bottom_px": args.bottom,
        "border_left_px": args.left,
        "border_right_px": args.right,
        "master_width_px": master.width,
        "master_height_px": master.height,
        "detection_method": "manual",
        "confidence": 1.0,
        "needs_review": False,
    }

    zone_map_path.write_text(json.dumps(zone_map, indent=2))
    logger.info("Zone map updated for design %s: %s", design_id, zone_map)


def cmd_cleanup_cache(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Delete cached files older than max_age_days."""
    max_age = args.max_age_days
    deleted = cleanup_cache(config.cache_dir, max_age)
    logger.info("Deleted %d stale cache file(s)", len(deleted))
    for p in deleted:
        print(f"  Deleted: {p}")


def cmd_list(args: argparse.Namespace, config: PipelineConfig) -> None:
    """List all designs and their status."""
    masters_dir = config.masters_dir

    if not masters_dir.exists():
        print("No designs found.")
        return

    designs = sorted(d.name for d in masters_dir.iterdir() if d.is_dir())
    if not designs:
        print("No designs found.")
        return

    for design_id in designs:
        design_dir = masters_dir / design_id
        has_master = (design_dir / "master.tiff").exists()
        has_zones = (design_dir / "zone_map.json").exists()
        has_thumb = (design_dir / "thumbnail.png").exists()

        status_parts = []
        if has_master:
            status_parts.append("master")
        if has_thumb:
            status_parts.append("thumbnail")

        needs_review = False
        if has_zones:
            zone_data = json.loads((design_dir / "zone_map.json").read_text())
            needs_review = zone_data.get("needs_review", False)
            confidence = zone_data.get("confidence", 0)
            method = zone_data.get("detection_method", "unknown")
            status_parts.append(f"zones({method}, conf={confidence:.2f})")

        if needs_review:
            status_parts.append("NEEDS REVIEW")

        print(f"  {design_id}: {', '.join(status_parts)}")


def cmd_validate(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Validate a design's master against its original."""
    design_id = args.design_id
    master_path = config.masters_dir / design_id / "master.tiff"
    archive_path = config.archive_dir / design_id

    if not master_path.exists():
        logger.error("Master TIFF not found for design %s", design_id)
        sys.exit(1)

    # Find the original in archive
    originals = list(archive_path.glob("original.*")) if archive_path.exists() else []
    if not originals:
        logger.error("Original file not found in archive for design %s", design_id)
        sys.exit(1)

    result = validate_flatten(
        originals[0], master_path,
        ssim_threshold=config.validation_ssim_threshold,
    )

    print(f"  SSIM Score: {result['ssim_score']}")
    print(f"  Passed: {result['passed']}")
    print(f"  Threshold: {result['threshold']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rug design pipeline — ingest, scale, and manage rug designs.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./pipeline_data"),
        help="Base directory for pipeline data (default: ./pipeline_data)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a new PSD design")
    p_ingest.add_argument("--psd", required=True, help="Path to the PSD file")
    p_ingest.add_argument("--design-id", required=True, help="Design identifier")

    # scale
    p_scale = subparsers.add_parser("scale", help="Scale a design to a rug size")
    p_scale.add_argument("--design-id", required=True)
    p_scale.add_argument("--width", type=float, required=True, help="Width in feet")
    p_scale.add_argument("--height", type=float, required=True, help="Height in feet")
    p_scale.add_argument("--dpi", type=int, default=150)

    # set-zones
    p_zones = subparsers.add_parser("set-zones", help="Manually set border zones")
    p_zones.add_argument("--design-id", required=True)
    p_zones.add_argument("--top", type=int, required=True)
    p_zones.add_argument("--bottom", type=int, required=True)
    p_zones.add_argument("--left", type=int, required=True)
    p_zones.add_argument("--right", type=int, required=True)

    # cleanup-cache
    p_cleanup = subparsers.add_parser("cleanup-cache", help="Delete stale cache files")
    p_cleanup.add_argument("--max-age-days", type=int, default=30)

    # list
    subparsers.add_parser("list", help="List all designs")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate a design's master")
    p_validate.add_argument("--design-id", required=True)

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PipelineConfig(base_dir=args.data_dir)

    commands = {
        "ingest": cmd_ingest,
        "scale": cmd_scale,
        "set-zones": cmd_set_zones,
        "cleanup-cache": cmd_cleanup_cache,
        "list": cmd_list,
        "validate": cmd_validate,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
