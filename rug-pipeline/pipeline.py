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
import tempfile
from pathlib import Path

# Add project root to path so modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from detection.border_detect import detect_borders
from ingest.flatten import flatten_psd
from ingest.validate import validate_flatten
from scaling.cache import check_cache, cleanup_cache, write_cache
from scaling.layered_psd import scale_layered_psd
from scaling.nine_slice import nine_slice_scale
from scaling.resize import lanczos_resize, make_esrgan_resize, needs_upscale
from storage.backend import StorageBackend, create_backend

logger = logging.getLogger("pipeline")


def _get_storage(config: PipelineConfig) -> tuple[StorageBackend, str, str]:
    """Create storage backend and return (backend, masters_bucket, archive_bucket)."""
    backend = create_backend(
        config.storage_backend,
        region=config.s3_region,
        glacier_transition_days=config.s3_glacier_transition_days,
    )

    if config.storage_backend == "s3":
        masters_bucket = config.s3_masters_bucket
        archive_bucket = config.s3_archive_bucket
    else:
        masters_bucket = str(config.masters_dir)
        archive_bucket = str(config.archive_dir)

    return backend, masters_bucket, archive_bucket


def cmd_ingest(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Ingest a PSD: flatten, detect borders, validate, archive."""
    psd_path = Path(args.psd)
    design_id = args.design_id

    if not psd_path.exists():
        logger.error("PSD file not found: %s", psd_path)
        sys.exit(1)

    config.ensure_dirs()
    storage, masters_bucket, archive_bucket = _get_storage(config)

    # Always flatten to local temp first (pyvips needs local files)
    with tempfile.TemporaryDirectory(prefix="rug_ingest_") as tmpdir:
        tmpdir = Path(tmpdir)
        master_tiff = tmpdir / "master.tiff"
        thumbnail = tmpdir / "thumbnail.png"

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

        # Upload outputs to storage backend
        logger.info("Uploading master, thumbnail, and zone map...")
        storage.put(master_tiff, f"{design_id}/master.tiff", masters_bucket)
        storage.put(thumbnail, f"{design_id}/thumbnail.png", masters_bucket)
        storage.write_json(zone_map, f"{design_id}/zone_map.json", masters_bucket)
        logger.info("Uploaded to masters storage: %s", masters_bucket)

    # Step 4: Archive original PSD
    logger.info("Step 4/4: Archiving original PSD...")
    archive_key = f"{design_id}/original{psd_path.suffix}"
    storage.put(psd_path, archive_key, archive_bucket)

    # Configure Glacier lifecycle if using S3
    if config.storage_backend == "s3":
        storage.configure_glacier_lifecycle(archive_bucket)

    # Remove original PSD after successful archive
    psd_path.unlink()
    logger.info("Archived PSD and removed local copy")

    logger.info("Ingest complete for design %s", design_id)


def cmd_scale(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Generate a print-ready file at a specific rug size."""
    design_id = args.design_id
    width_ft = args.width
    height_ft = args.height
    dpi = args.dpi

    config.ensure_dirs()
    storage, masters_bucket, _ = _get_storage(config)

    # Check local cache first
    cached = check_cache(config.cache_dir, design_id, width_ft, height_ft, dpi, config.cache_max_age_days)
    if cached:
        logger.info("Using cached file: %s", cached)
        print(str(cached))
        return

    # Download master and zone map to local temp (pyvips needs local files)
    with tempfile.TemporaryDirectory(prefix="rug_scale_") as tmpdir:
        tmpdir = Path(tmpdir)
        local_master = tmpdir / "master.tiff"
        zone_map_key = f"{design_id}/zone_map.json"
        master_key = f"{design_id}/master.tiff"

        if not storage.exists(master_key, masters_bucket):
            logger.error("Master TIFF not found for design %s", design_id)
            sys.exit(1)

        if not storage.exists(zone_map_key, masters_bucket):
            logger.error("Zone map not found for design %s", design_id)
            sys.exit(1)

        storage.get(master_key, masters_bucket, local_master)
        zone_map = storage.read_json(zone_map_key, masters_bucket)

        import pyvips
        master = pyvips.Image.new_from_file(str(local_master))

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
                models_dir=config.realesrgan_models_dir,
            )
        else:
            resize_fn = lanczos_resize

        # 9-slice scale
        result = nine_slice_scale(
            master, zone_map, target_w, target_h,
            resize_fn=resize_fn,
            seam_threshold=config.seam_threshold,
        )

        # Force pixels into RAM before temp dir is cleaned up
        # (pyvips uses lazy I/O — the image still references the temp file)
        result = result.copy_memory()

    # Save to local cache (outside temp dir — result is in memory)
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
    storage, masters_bucket, _ = _get_storage(config)

    master_key = f"{design_id}/master.tiff"
    if not storage.exists(master_key, masters_bucket):
        logger.error("Master TIFF not found for design %s", design_id)
        sys.exit(1)

    # Get master dimensions — download to temp to read with pyvips
    with tempfile.TemporaryDirectory(prefix="rug_zones_") as tmpdir:
        local_master = Path(tmpdir) / "master.tiff"
        storage.get(master_key, masters_bucket, local_master)

        import pyvips
        master = pyvips.Image.new_from_file(str(local_master))
        master_w, master_h = master.width, master.height

    zone_map = {
        "design_id": design_id,
        "border_top_px": args.top,
        "border_bottom_px": args.bottom,
        "border_left_px": args.left,
        "border_right_px": args.right,
        "master_width_px": master_w,
        "master_height_px": master_h,
        "detection_method": "manual",
        "confidence": 1.0,
        "needs_review": False,
    }

    storage.write_json(zone_map, f"{design_id}/zone_map.json", masters_bucket)
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
    storage, masters_bucket, _ = _get_storage(config)

    # List design IDs by finding zone_map.json files
    all_keys = storage.list_keys("", masters_bucket)
    design_ids = sorted({k.split("/")[0] for k in all_keys if "/" in k})

    if not design_ids:
        print("No designs found.")
        return

    for design_id in design_ids:
        has_master = storage.exists(f"{design_id}/master.tiff", masters_bucket)
        has_thumb = storage.exists(f"{design_id}/thumbnail.png", masters_bucket)
        has_zones = storage.exists(f"{design_id}/zone_map.json", masters_bucket)

        status_parts = []
        if has_master:
            status_parts.append("master")
        if has_thumb:
            status_parts.append("thumbnail")

        needs_review = False
        if has_zones:
            zone_data = storage.read_json(f"{design_id}/zone_map.json", masters_bucket)
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
    storage, masters_bucket, archive_bucket = _get_storage(config)

    master_key = f"{design_id}/master.tiff"
    if not storage.exists(master_key, masters_bucket):
        logger.error("Master TIFF not found for design %s", design_id)
        sys.exit(1)

    # Find the original in archive
    archive_keys = storage.list_keys(f"{design_id}/", archive_bucket)
    originals = [k for k in archive_keys if "/original." in k]
    if not originals:
        logger.error("Original file not found in archive for design %s", design_id)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="rug_validate_") as tmpdir:
        tmpdir = Path(tmpdir)
        local_master = tmpdir / "master.tiff"
        local_original = tmpdir / Path(originals[0]).name

        storage.get(master_key, masters_bucket, local_master)
        storage.get(originals[0], archive_bucket, local_original)

        result = validate_flatten(
            local_original, local_master,
            ssim_threshold=config.validation_ssim_threshold,
        )

    print(f"  SSIM Score: {result['ssim_score']}")
    print(f"  Passed: {result['passed']}")
    print(f"  Threshold: {result['threshold']}")


def cmd_scale_psd(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Scale a layered PSD to a new rug size, preserving layers."""
    width_ft = args.width
    height_ft = args.height
    dpi = args.dpi
    target_w = int(width_ft * 12 * dpi)
    target_h = int(height_ft * 12 * dpi)

    storage, masters_bucket, archive_bucket = _get_storage(config)

    # Get the PSD file — either from --psd flag or archive
    if args.psd:
        psd_path = Path(args.psd)
        if not psd_path.exists():
            logger.error("PSD file not found: %s", psd_path)
            sys.exit(1)
    else:
        if not args.design_id:
            logger.error("Must provide either --psd or --design-id")
            sys.exit(1)
        # Download from archive
        archive_keys = storage.list_keys(f"{args.design_id}/", archive_bucket)
        originals = [k for k in archive_keys if "/original." in k]
        if not originals:
            logger.error("Original PSD not found in archive for design %s", args.design_id)
            sys.exit(1)
        psd_path = Path(tempfile.mkdtemp(prefix="rug_psd_")) / Path(originals[0]).name
        storage.get(originals[0], archive_bucket, psd_path)

    # Get zone map — either from --zone-map flag or storage
    if args.zone_map:
        zone_map = json.loads(Path(args.zone_map).read_text())
    elif args.design_id:
        zone_map = storage.read_json(f"{args.design_id}/zone_map.json", masters_bucket)
    else:
        logger.error("Must provide either --zone-map or --design-id")
        sys.exit(1)

    # Output path
    design_label = args.design_id or psd_path.stem
    output_dir = config.cache_dir / design_label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{width_ft}x{height_ft}_{dpi}dpi_layered.psd"

    logger.info(
        "Scaling layered PSD to %sx%s ft (%dx%d px at %d DPI)",
        width_ft, height_ft, target_w, target_h, dpi,
    )

    scale_layered_psd(psd_path, output_path, zone_map, target_w, target_h)

    logger.info("Output saved: %s", output_path)
    print(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rug design pipeline — ingest, scale, and manage rug designs.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./pipeline_data"),
        help="Base directory for pipeline data (default: ./pipeline_data)",
    )
    parser.add_argument(
        "--storage", choices=["local", "s3"], default=None,
        help="Storage backend (overrides config default)",
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

    # scale-psd
    p_scale_psd = subparsers.add_parser("scale-psd", help="Scale a layered PSD preserving layers")
    p_scale_psd.add_argument("--design-id", help="Design ID (fetches PSD from archive)")
    p_scale_psd.add_argument("--psd", help="Path to PSD file (alternative to --design-id)")
    p_scale_psd.add_argument("--zone-map", help="Path to zone map JSON (alternative to --design-id)")
    p_scale_psd.add_argument("--width", type=float, required=True, help="Width in feet")
    p_scale_psd.add_argument("--height", type=float, required=True, help="Height in feet")
    p_scale_psd.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PipelineConfig(base_dir=args.data_dir)
    if args.storage:
        config.storage_backend = args.storage

    commands = {
        "ingest": cmd_ingest,
        "scale": cmd_scale,
        "scale-psd": cmd_scale_psd,
        "set-zones": cmd_set_zones,
        "cleanup-cache": cmd_cleanup_cache,
        "list": cmd_list,
        "validate": cmd_validate,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
