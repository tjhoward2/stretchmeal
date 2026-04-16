"""Resize orchestration: Lanczos and Real-ESRGAN wrapper.

Provides resize functions compatible with the 9-slice engine's ResizeFn
callback signature. Falls back to Lanczos if Real-ESRGAN is unavailable.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyvips

logger = logging.getLogger(__name__)


def lanczos_resize(image: pyvips.Image, target_w: int, target_h: int) -> pyvips.Image:
    """Resize using Lanczos3 resampling."""
    if target_w == image.width and target_h == image.height:
        return image
    h_scale = target_w / image.width
    v_scale = target_h / image.height
    return image.resize(h_scale, vscale=v_scale, kernel="lanczos3")


def make_esrgan_resize(
    binary: str = "realesrgan-ncnn-vulkan",
    scale: int = 4,
    model: str = "realesrgan-x4plus",
) -> callable:
    """Create a resize function that uses Real-ESRGAN for upscaling.

    Returns a function with the ResizeFn signature. If the binary is not
    found on PATH, returns a function that logs a warning and falls back
    to Lanczos.

    Args:
        binary: Name or path of the realesrgan-ncnn-vulkan binary.
        scale: Upscale factor (default 4x).
        model: Model name for Real-ESRGAN.

    Returns:
        A resize function (image, target_w, target_h) -> image.
    """
    if not shutil.which(binary):
        logger.warning(
            "Real-ESRGAN binary '%s' not found on PATH. "
            "AI upscaling unavailable — will fall back to Lanczos.",
            binary,
        )
        return lanczos_resize

    def esrgan_resize(image: pyvips.Image, target_w: int, target_h: int) -> pyvips.Image:
        """Upscale with Real-ESRGAN, then Lanczos-resize to exact target."""
        # If downscaling or same size, just use Lanczos
        if target_w <= image.width and target_h <= image.height:
            return lanczos_resize(image, target_w, target_h)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Save input as PNG for Real-ESRGAN
            image.pngsave(str(input_path))

            # Run Real-ESRGAN
            cmd = [
                binary,
                "-i", str(input_path),
                "-o", str(output_path),
                "-s", str(scale),
                "-n", model,
            ]

            logger.info("Running Real-ESRGAN: %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    logger.error("Real-ESRGAN failed: %s", result.stderr)
                    logger.warning("Falling back to Lanczos upscale")
                    return lanczos_resize(image, target_w, target_h)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.error("Real-ESRGAN error: %s", e)
                logger.warning("Falling back to Lanczos upscale")
                return lanczos_resize(image, target_w, target_h)

            # Load the upscaled result
            upscaled = pyvips.Image.new_from_file(str(output_path))

            # Final Lanczos resize to exact target dimensions
            return lanczos_resize(upscaled, target_w, target_h)

    return esrgan_resize


def needs_upscale(master_w: int, master_h: int, target_w: int, target_h: int) -> bool:
    """Check if the target requires upscaling in either dimension."""
    return target_w > master_w or target_h > master_h
