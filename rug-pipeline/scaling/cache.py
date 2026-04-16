"""Cache management for scaled rug output files.

Handles reading, writing, staleness checks, and cleanup of cached
print-ready TIFF files.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_path(
    cache_dir: Path,
    design_id: str,
    width_ft: float,
    height_ft: float,
    dpi: int,
) -> Path:
    """Build the cache file path for a specific design/size combination.

    Returns:
        Path like cache_dir/{design_id}/{width}x{height}_{dpi}dpi.tiff
    """
    filename = f"{width_ft}x{height_ft}_{dpi}dpi.tiff"
    return cache_dir / design_id / filename


def metadata_path(tiff_path: Path) -> Path:
    """Get the metadata JSON path for a cached TIFF."""
    return tiff_path.with_suffix(".json")


def check_cache(
    cache_dir: Path,
    design_id: str,
    width_ft: float,
    height_ft: float,
    dpi: int,
    max_age_days: int = 30,
) -> Path | None:
    """Check if a valid (non-stale) cached file exists.

    Args:
        cache_dir: Base cache directory.
        design_id: Design identifier.
        width_ft: Target width in feet.
        height_ft: Target height in feet.
        dpi: Target DPI.
        max_age_days: Max age in days before considering stale.

    Returns:
        Path to cached file if valid, None otherwise.
    """
    path = cache_path(cache_dir, design_id, width_ft, height_ft, dpi)

    if not path.exists():
        return None

    # Check staleness
    age_seconds = time.time() - path.stat().st_mtime
    age_days = age_seconds / 86400

    if age_days > max_age_days:
        logger.info(
            "Cache hit for %s at %sx%s %ddpi but stale (%.1f days old)",
            design_id, width_ft, height_ft, dpi, age_days,
        )
        return None

    logger.info(
        "Cache hit for %s at %sx%s %ddpi (%.1f days old)",
        design_id, width_ft, height_ft, dpi, age_days,
    )
    return path


def write_cache(
    image,  # pyvips.Image — not type-hinted to avoid import
    cache_dir: Path,
    design_id: str,
    width_ft: float,
    height_ft: float,
    dpi: int,
    needs_approval: bool = False,
) -> Path:
    """Save a scaled image to the cache.

    Args:
        image: pyvips Image to save.
        cache_dir: Base cache directory.
        design_id: Design identifier.
        width_ft: Target width in feet.
        height_ft: Target height in feet.
        dpi: Target DPI.
        needs_approval: Whether this output needs quality approval.

    Returns:
        Path to the saved cache file.
    """
    path = cache_path(cache_dir, design_id, width_ft, height_ft, dpi)
    path.parent.mkdir(parents=True, exist_ok=True)

    image.tiffsave(str(path), compression="lzw")
    logger.info("Cached output: %s", path)

    # Write metadata
    meta = {
        "design_id": design_id,
        "width_ft": width_ft,
        "height_ft": height_ft,
        "dpi": dpi,
        "width_px": image.width,
        "height_px": image.height,
        "needs_approval": needs_approval,
    }
    meta_path = metadata_path(path)
    meta_path.write_text(json.dumps(meta, indent=2))

    return path


def cleanup_cache(cache_dir: Path, max_age_days: int = 30) -> list[Path]:
    """Delete cached files older than max_age_days.

    Args:
        cache_dir: Base cache directory.
        max_age_days: Max age in days.

    Returns:
        List of deleted file paths.
    """
    deleted = []
    now = time.time()
    max_age_seconds = max_age_days * 86400

    if not cache_dir.exists():
        return deleted

    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue

        age = now - path.stat().st_mtime
        if age > max_age_seconds:
            logger.info("Deleting stale cache file: %s (%.1f days old)", path, age / 86400)
            path.unlink()
            deleted.append(path)

    # Clean up empty directories
    for dirpath in sorted(cache_dir.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()
            logger.debug("Removed empty directory: %s", dirpath)

    return deleted
