"""Automatic border detection for rug designs.

Scans inward from each edge using color gradient analysis and Canny edge
detection to locate the border/interior boundary. Produces a zone map
compatible with the 9-slice scaling engine.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import pyvips

logger = logging.getLogger(__name__)

# Defaults (can be overridden via config)
_DETECTION_WIDTH = 2000
_CANNY_LOW = 50
_CANNY_HIGH = 150
_GRADIENT_WEIGHT = 0.6
_EDGE_WEIGHT = 0.4
_CONFIDENCE_THRESHOLD = 0.7


def detect_borders(
    image: pyvips.Image,
    design_id: str,
    detection_width: int = _DETECTION_WIDTH,
    canny_low: int = _CANNY_LOW,
    canny_high: int = _CANNY_HIGH,
    gradient_weight: float = _GRADIENT_WEIGHT,
    edge_weight: float = _EDGE_WEIGHT,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> dict:
    """Detect border regions in a rug design image.

    Args:
        image: Full-resolution pyvips master image.
        design_id: Identifier for the design (included in output).
        detection_width: Width to downscale to for faster processing.
        canny_low: Canny edge detector low threshold.
        canny_high: Canny edge detector high threshold.
        gradient_weight: Weight for color gradient signal (0-1).
        edge_weight: Weight for Canny edge signal (0-1).
        confidence_threshold: Below this, flag for manual review.

    Returns:
        Zone map dict with border pixel widths (at master resolution),
        confidence score, and needs_review flag.
    """
    master_w = image.width
    master_h = image.height
    scale = detection_width / master_w

    # Downscale for processing
    thumb = image.resize(scale, kernel="lanczos3")
    thumb_w = thumb.width
    thumb_h = thumb.height

    # Convert pyvips to numpy for OpenCV processing
    np_img = _vips_to_numpy(thumb)

    # Convert to grayscale for edge detection
    if len(np_img.shape) == 3 and np_img.shape[2] >= 3:
        gray = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img if len(np_img.shape) == 2 else np_img[:, :, 0]

    # Run Canny edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Detect each border
    borders = {}
    confidences = []

    for side in ("top", "bottom", "left", "right"):
        pos, conf = _detect_one_border(
            np_img, gray, edges, side,
            gradient_weight=gradient_weight,
            edge_weight=edge_weight,
        )
        borders[side] = pos
        confidences.append(conf)

    # Scale border positions back to master resolution
    border_top = int(round(borders["top"] / scale))
    border_bottom = int(round(borders["bottom"] / scale))
    border_left = int(round(borders["left"] / scale))
    border_right = int(round(borders["right"] / scale))

    # Overall confidence
    if all(b == 0 for b in [border_top, border_bottom, border_left, border_right]):
        # Borderless design — perfectly valid
        confidence = 1.0
    else:
        confidence = float(np.mean(confidences)) if confidences else 0.0

    needs_review = confidence < confidence_threshold

    zone_map = {
        "design_id": str(design_id),
        "border_top_px": border_top,
        "border_bottom_px": border_bottom,
        "border_left_px": border_left,
        "border_right_px": border_right,
        "master_width_px": master_w,
        "master_height_px": master_h,
        "detection_method": "auto",
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
    }

    logger.info(
        "Border detection for design %s: top=%d, bottom=%d, left=%d, right=%d "
        "(confidence=%.2f, needs_review=%s)",
        design_id,
        border_top, border_bottom, border_left, border_right,
        confidence, needs_review,
    )

    return zone_map


def _detect_one_border(
    color_img: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    side: str,
    gradient_weight: float = 0.6,
    edge_weight: float = 0.4,
    max_scan_fraction: float = 0.4,
) -> tuple[int, float]:
    """Detect one border boundary by scanning inward from an edge.

    Returns:
        Tuple of (border_width_in_detection_pixels, confidence).
    """
    h, w = gray.shape
    max_scan = int(h * max_scan_fraction) if side in ("top", "bottom") else int(w * max_scan_fraction)

    if max_scan < 3:
        return 0, 0.0

    # Compute per-row/column mean colors and gradient magnitudes
    if side == "top":
        # Scan rows from top down
        means = np.array([color_img[r, :].mean(axis=0) for r in range(max_scan)])
        edge_projection = edges[:max_scan, :].mean(axis=1)
    elif side == "bottom":
        means = np.array([color_img[h - 1 - r, :].mean(axis=0) for r in range(max_scan)])
        edge_projection = edges[h - max_scan:, :][::-1].mean(axis=1)
    elif side == "left":
        means = np.array([color_img[:, c].mean(axis=0) for c in range(max_scan)])
        edge_projection = edges[:, :max_scan].mean(axis=0)
    else:  # right
        means = np.array([color_img[:, w - 1 - c].mean(axis=0) for c in range(max_scan)])
        edge_projection = edges[:, w - max_scan:][:, ::-1].mean(axis=0)

    # Color gradient: difference between consecutive row/col means
    if len(means.shape) > 1:
        diffs = np.linalg.norm(np.diff(means, axis=0), axis=1)
    else:
        diffs = np.abs(np.diff(means))

    if len(diffs) == 0:
        return 0, 0.0

    # Normalize both signals to 0-1
    grad_max = diffs.max()
    gradient_signal = diffs / grad_max if grad_max > 0 else diffs

    edge_max = edge_projection.max()
    edge_signal = edge_projection / edge_max if edge_max > 0 else edge_projection

    # Align lengths (edge_projection may be 1 longer)
    min_len = min(len(gradient_signal), len(edge_signal))
    gradient_signal = gradient_signal[:min_len]
    edge_signal = edge_signal[:min_len]

    # Combined signal
    combined = gradient_weight * gradient_signal + edge_weight * edge_signal

    # Find the strongest transition
    peak_idx = int(np.argmax(combined))
    peak_val = combined[peak_idx]

    # Confidence based on peak prominence
    mean_val = combined.mean()
    if mean_val > 0:
        prominence = (peak_val - mean_val) / peak_val
        confidence = min(1.0, prominence)
    else:
        confidence = 0.0

    # If the peak is too weak, report no border
    if peak_val < 0.1:
        return 0, 0.0

    # The border width is the position of the peak (in detection pixels)
    # +1 because diff shifts indices by 1
    border_width = peak_idx + 1

    return border_width, confidence


def _vips_to_numpy(image: pyvips.Image) -> np.ndarray:
    """Convert a pyvips Image to a numpy array."""
    mem = image.write_to_memory()
    return np.ndarray(
        buffer=mem,
        dtype=np.uint8,
        shape=(image.height, image.width, image.bands),
    )
