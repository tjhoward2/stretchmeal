"""PSD flattening module.

Loads the composite view of a PSD file using pyvips and saves it as
a TIFF with LZW compression. Also generates a low-res thumbnail.
"""

import logging
from pathlib import Path

import pyvips

logger = logging.getLogger(__name__)


def flatten_psd(psd_path: Path, output_tiff: Path, thumbnail_path: Path,
                thumbnail_width: int = 1000) -> pyvips.Image:
    """Flatten a PSD file to a TIFF master and generate a thumbnail.

    Args:
        psd_path: Path to the source PSD file.
        output_tiff: Path where the flattened TIFF will be saved.
        thumbnail_path: Path where the PNG thumbnail will be saved.
        thumbnail_width: Target width for the thumbnail in pixels.

    Returns:
        The loaded pyvips image (for further processing like border detection).

    Raises:
        FileNotFoundError: If the PSD file doesn't exist.
        pyvips.Error: If pyvips can't read the file.
    """
    psd_path = Path(psd_path)
    if not psd_path.exists():
        raise FileNotFoundError(f"PSD file not found: {psd_path}")

    logger.info("Flattening PSD: %s", psd_path)

    # Load the composite (flattened) view — pyvips reads this by default.
    # Use sequential access for streaming large files.
    image = pyvips.Image.new_from_file(str(psd_path), access="sequential")

    logger.info(
        "Loaded image: %dx%d, %d bands, format=%s",
        image.width, image.height, image.bands, image.format,
    )

    # Ensure output directories exist
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as TIFF with LZW compression (strip tiling so we can re-read linearly)
    image.tiffsave(str(output_tiff), compression="lzw", tile=False)
    logger.info("Saved master TIFF: %s", output_tiff)

    # Re-open the TIFF (random access) for thumbnail generation
    master = pyvips.Image.new_from_file(str(output_tiff))

    # Generate thumbnail from the saved master
    thumb_scale = thumbnail_width / master.width
    thumbnail = master.resize(thumb_scale, kernel="lanczos3")
    thumbnail.pngsave(str(thumbnail_path))
    logger.info("Saved thumbnail: %s (%dx%d)", thumbnail_path, thumbnail.width, thumbnail.height)

    return master
