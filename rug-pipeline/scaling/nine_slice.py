"""9-slice scaling engine for border-aware rug resizing.

Splits a rug image into 9 regions based on a zone map, resizes each
region according to its role (corners fixed, borders scaled in one axis,
interior scaled freely), and reassembles into the target dimensions.
"""

import logging
from typing import Callable, Optional

import numpy as np
import pyvips

logger = logging.getLogger(__name__)

# Type alias for resize callbacks
ResizeFn = Callable[[pyvips.Image, int, int], pyvips.Image]


def default_resize(image: pyvips.Image, target_w: int, target_h: int) -> pyvips.Image:
    """Resize using Lanczos3 resampling (default for downscale/moderate upscale)."""
    if target_w == image.width and target_h == image.height:
        return image
    h_scale = target_w / image.width
    v_scale = target_h / image.height
    return image.resize(h_scale, vscale=v_scale, kernel="lanczos3")


def nine_slice_scale(
    image: pyvips.Image,
    zone_map: dict,
    target_w: int,
    target_h: int,
    resize_fn: Optional[ResizeFn] = None,
    seam_threshold: float = 5.0,
) -> pyvips.Image:
    """Scale a rug image using 9-slice border-aware logic.

    Args:
        image: Source pyvips image (the master TIFF).
        zone_map: Dict with border_top_px, border_bottom_px, border_left_px,
                  border_right_px, master_width_px, master_height_px.
        target_w: Target width in pixels.
        target_h: Target height in pixels.
        resize_fn: Optional callback for resizing individual regions.
                   Signature: (image, target_w, target_h) -> image.
                   Defaults to Lanczos3 resize.
        seam_threshold: Max avg color diff (0-255) along internal seams
                        before logging a warning.

    Returns:
        The assembled pyvips image at target dimensions.
    """
    if resize_fn is None:
        resize_fn = default_resize

    bt = zone_map.get("border_top_px", 0)
    bb = zone_map.get("border_bottom_px", 0)
    bl = zone_map.get("border_left_px", 0)
    br = zone_map.get("border_right_px", 0)

    src_w = image.width
    src_h = image.height

    # Calculate target interior size
    interior_w = target_w - bl - br
    interior_h = target_h - bt - bb

    # Fallback: if borders are too large for target, do uniform scale
    if interior_w < 100 or interior_h < 100:
        logger.warning(
            "Target interior too small (%dx%d px). "
            "Falling back to uniform scaling of entire image.",
            interior_w,
            interior_h,
        )
        return resize_fn(image, target_w, target_h)

    # Source interior dimensions
    src_interior_w = src_w - bl - br
    src_interior_h = src_h - bt - bb

    if src_interior_w <= 0 or src_interior_h <= 0:
        logger.warning(
            "Source border zones exceed image dimensions. "
            "Falling back to uniform scaling."
        )
        return resize_fn(image, target_w, target_h)

    # --- Crop 9 regions ---
    # Corners (never resized)
    corner_tl = image.crop(0, 0, bl, bt) if bl > 0 and bt > 0 else None
    corner_tr = image.crop(src_w - br, 0, br, bt) if br > 0 and bt > 0 else None
    corner_bl = image.crop(0, src_h - bb, bl, bb) if bl > 0 and bb > 0 else None
    corner_br = image.crop(src_w - br, src_h - bb, br, bb) if br > 0 and bb > 0 else None

    # Border strips (scaled in one axis)
    strip_top = image.crop(bl, 0, src_interior_w, bt) if bt > 0 else None
    strip_bottom = image.crop(bl, src_h - bb, src_interior_w, bb) if bb > 0 else None
    strip_left = image.crop(0, bt, bl, src_interior_h) if bl > 0 else None
    strip_right = image.crop(src_w - br, bt, br, src_interior_h) if br > 0 else None

    # Interior field
    interior = image.crop(bl, bt, src_interior_w, src_interior_h)

    # --- Resize pieces ---
    # Top/bottom strips: stretch width to interior_w, keep original height
    if strip_top is not None:
        strip_top = resize_fn(strip_top, interior_w, bt)
    if strip_bottom is not None:
        strip_bottom = resize_fn(strip_bottom, interior_w, bb)

    # Left/right strips: stretch height to interior_h, keep original width
    if strip_left is not None:
        strip_left = resize_fn(strip_left, bl, interior_h)
    if strip_right is not None:
        strip_right = resize_fn(strip_right, br, interior_h)

    # Interior: resize to target interior dimensions
    interior = resize_fn(interior, interior_w, interior_h)

    # --- Assemble on canvas ---
    # Create a blank canvas matching target dimensions and band count
    canvas = pyvips.Image.black(target_w, target_h, bands=image.bands)
    if image.interpretation == "srgb" or image.bands >= 3:
        canvas = canvas.copy(interpretation=image.interpretation)

    # Place pieces using insert
    # Corners
    if corner_tl is not None:
        canvas = canvas.insert(corner_tl, 0, 0)
    if corner_tr is not None:
        canvas = canvas.insert(corner_tr, target_w - br, 0)
    if corner_bl is not None:
        canvas = canvas.insert(corner_bl, 0, target_h - bb)
    if corner_br is not None:
        canvas = canvas.insert(corner_br, target_w - br, target_h - bb)

    # Border strips
    if strip_top is not None:
        canvas = canvas.insert(strip_top, bl, 0)
    if strip_bottom is not None:
        canvas = canvas.insert(strip_bottom, bl, target_h - bb)
    if strip_left is not None:
        canvas = canvas.insert(strip_left, 0, bt)
    if strip_right is not None:
        canvas = canvas.insert(strip_right, target_w - br, bt)

    # Interior
    canvas = canvas.insert(interior, bl, bt)

    # --- Seam check ---
    _check_seams(canvas, bl, br, bt, bb, target_w, target_h, seam_threshold)

    return canvas


def _check_seams(
    canvas: pyvips.Image,
    bl: int,
    br: int,
    bt: int,
    bb: int,
    w: int,
    h: int,
    threshold: float,
) -> None:
    """Check pixel continuity along internal seam lines.

    Samples 1px strips on each side of each internal boundary and
    compares mean color values. Logs warnings if differences exceed threshold.
    """
    seam_lines = []

    # Left border / interior boundary (vertical line at x=bl)
    if bl > 0 and bl < w - 1:
        seam_lines.append(("left-interior", bl))

    # Right border / interior boundary (vertical line at x=w-br)
    if br > 0 and w - br > 0:
        seam_lines.append(("right-interior", w - br))

    for name, x in seam_lines:
        if x <= 0 or x >= w:
            continue
        left_strip = canvas.crop(x - 1, 0, 1, h)
        right_strip = canvas.crop(x, 0, 1, h)
        diff = (left_strip - right_strip).abs().avg()
        if diff > threshold:
            logger.warning(
                "Seam '%s' at x=%d: avg color diff %.2f exceeds threshold %.1f",
                name, x, diff, threshold,
            )

    # Horizontal seam lines
    h_seam_lines = []
    if bt > 0 and bt < h - 1:
        h_seam_lines.append(("top-interior", bt))
    if bb > 0 and h - bb > 0:
        h_seam_lines.append(("bottom-interior", h - bb))

    for name, y in h_seam_lines:
        if y <= 0 or y >= h:
            continue
        top_strip = canvas.crop(0, y - 1, w, 1)
        bottom_strip = canvas.crop(0, y, w, 1)
        diff = (top_strip - bottom_strip).abs().avg()
        if diff > threshold:
            logger.warning(
                "Seam '%s' at y=%d: avg color diff %.2f exceeds threshold %.1f",
                name, y, diff, threshold,
            )
