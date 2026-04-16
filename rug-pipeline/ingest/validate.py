"""Validation module for flattened masters.

Compares a low-res thumbnail of the flattened output against the original
PSD composite using SSIM (structural similarity) to catch flattening errors.
"""

import logging
from pathlib import Path

import numpy as np
import pyvips
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def validate_flatten(
    original_path: Path,
    flattened_path: Path,
    comparison_width: int = 500,
    ssim_threshold: float = 0.95,
) -> dict:
    """Validate a flattened TIFF against its original PSD.

    Generates low-res versions of both and compares via SSIM.

    Args:
        original_path: Path to the original PSD file.
        flattened_path: Path to the flattened TIFF.
        comparison_width: Width to downscale both images for comparison.
        ssim_threshold: Minimum SSIM score to pass validation.

    Returns:
        Dict with ssim_score, passed, and needs_review fields.
    """
    logger.info("Validating flattened output against original")

    # Load both images
    original = pyvips.Image.new_from_file(str(original_path), access="sequential")
    flattened = pyvips.Image.new_from_file(str(flattened_path), access="sequential")

    # Downscale both to comparison size
    orig_thumb = _make_thumb(original, comparison_width)
    flat_thumb = _make_thumb(flattened, comparison_width)

    # Ensure same dimensions (they should be, but guard against rounding)
    min_h = min(orig_thumb.shape[0], flat_thumb.shape[0])
    min_w = min(orig_thumb.shape[1], flat_thumb.shape[1])
    orig_thumb = orig_thumb[:min_h, :min_w]
    flat_thumb = flat_thumb[:min_h, :min_w]

    # Ensure same number of channels (convert to 3-channel if needed)
    orig_thumb = _normalize_channels(orig_thumb)
    flat_thumb = _normalize_channels(flat_thumb)

    # Compute SSIM
    score = ssim(orig_thumb, flat_thumb, channel_axis=2, data_range=255)
    passed = bool(score >= ssim_threshold)

    result = {
        "ssim_score": round(float(score), 4),
        "passed": passed,
        "needs_review": not passed,
        "threshold": ssim_threshold,
    }

    if passed:
        logger.info("Validation passed: SSIM=%.4f (threshold=%.2f)", score, ssim_threshold)
    else:
        logger.warning(
            "Validation FAILED: SSIM=%.4f below threshold=%.2f — flagging for review",
            score, ssim_threshold,
        )

    return result


def _make_thumb(image: pyvips.Image, width: int) -> np.ndarray:
    """Downscale a pyvips image and convert to numpy."""
    scale = width / image.width
    thumb = image.resize(scale, kernel="lanczos3")
    mem = thumb.write_to_memory()
    return np.ndarray(
        buffer=mem,
        dtype=np.uint8,
        shape=(thumb.height, thumb.width, thumb.bands),
    )


def _normalize_channels(arr: np.ndarray) -> np.ndarray:
    """Normalize to 3-channel RGB."""
    if len(arr.shape) == 2:
        return np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 1:
        return np.concatenate([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    return arr
