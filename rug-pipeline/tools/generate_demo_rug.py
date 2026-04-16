#!/usr/bin/env python3
"""Generate a large synthetic rug image for demo purposes.

Creates a ~2.4 GB TIFF with a rug-like pattern.

Usage:
    python tools/generate_demo_rug.py [--output demo_rug.tiff]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyvips


def generate_demo_rug(output_path: str = "demo_rug.tiff"):
    width = 24000
    height = 33600
    border = 800

    estimated_gb = width * height * 3 / (1024 ** 3)
    print(f"Generating {width}x{height} rug image (~{estimated_gb:.1f} GB)...")
    print("This will take several minutes...")

    # Build full image in numpy (needs ~2.4 GB RAM)
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Interior: deep navy with texture ---
    print("  Painting interior...")
    interior = arr[border:height - border, border:width - border]
    iy, ix = np.mgrid[0:interior.shape[0], 0:interior.shape[1]]

    # Base navy
    interior[:, :, 0] = 20
    interior[:, :, 1] = 30
    interior[:, :, 2] = 75

    # Central medallion
    cy, cx = interior.shape[0] // 2, interior.shape[1] // 2
    dy = (iy - cy).astype(np.float32) / 5600
    dx = (ix - cx).astype(np.float32) / 4000
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Red center
    center_mask = dist < 0.3
    interior[center_mask, 0] = 160
    interior[center_mask, 1] = 35
    interior[center_mask, 2] = 30

    # Gold ring
    ring_mask = (dist >= 0.3) & (dist < 0.35)
    interior[ring_mask, 0] = 195
    interior[ring_mask, 1] = 155
    interior[ring_mask, 2] = 40

    # Lighter blue inner medallion
    inner_mask = (dist >= 0.35) & (dist < 0.7)
    interior[inner_mask, 0] = 30
    interior[inner_mask, 1] = 50
    interior[inner_mask, 2] = 110

    # Small repeating motifs in field
    field_x = (ix % 500)
    field_y = (iy % 500)
    motif = (np.abs(field_x - 250) + np.abs(field_y - 250)) < 40
    interior[motif, 0] = np.minimum(255, interior[motif, 0].astype(np.int16) + 25).astype(np.uint8)
    interior[motif, 1] = np.minimum(255, interior[motif, 1].astype(np.int16) + 15).astype(np.uint8)

    # Texture noise
    rng = np.random.default_rng(42)
    noise = rng.integers(-6, 7, interior.shape[:2], dtype=np.int8)
    for c in range(3):
        interior[:, :, c] = np.clip(interior[:, :, c].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    arr[border:height - border, border:width - border] = interior
    del interior, dist, dx, dy, iy, ix

    # --- Borders: crimson with gold bands ---
    print("  Painting borders...")

    # Top border
    top = arr[0:border, border:width - border]
    top[:, :] = [155, 30, 25]
    for band_y in [40, 100, 200, 350, border - 350, border - 200, border - 100, border - 40]:
        if band_y + 8 <= border:
            top[band_y:band_y + 8, :] = [200, 160, 45]
    # Repeating diamonds
    ty, tx = np.mgrid[0:top.shape[0], 0:top.shape[1]]
    diamond = (np.abs(tx % 300 - 150) + np.abs(ty % 300 - 150)) < 30
    top[diamond, 0] = np.minimum(255, top[diamond, 0].astype(np.int16) + 40).astype(np.uint8)
    arr[0:border, border:width - border] = top

    # Bottom border
    arr[height - border:, border:width - border] = arr[0:border, border:width - border][::-1]

    # Left border
    left = arr[border:height - border, 0:border]
    left[:, :] = [155, 30, 25]
    for band_x in [40, 100, 200, 350, border - 350, border - 200, border - 100, border - 40]:
        if band_x + 8 <= border:
            left[:, band_x:band_x + 8] = [200, 160, 45]
    ly, lx = np.mgrid[0:left.shape[0], 0:left.shape[1]]
    diamond_l = (np.abs(lx % 300 - 150) + np.abs(ly % 300 - 150)) < 30
    left[diamond_l, 0] = np.minimum(255, left[diamond_l, 0].astype(np.int16) + 40).astype(np.uint8)
    arr[border:height - border, 0:border] = left

    # Right border
    arr[border:height - border, width - border:] = arr[border:height - border, 0:border][:, ::-1]

    # --- Corners: gold with crosshatch ---
    print("  Painting corners...")
    corner = np.full((border, border, 3), [185, 145, 35], dtype=np.uint8)
    cy, cx = np.mgrid[0:border, 0:border]
    crosshatch = (cx % 60 < 3) | (cy % 60 < 3)
    corner[crosshatch] = [140, 100, 20]
    diamond_c = (np.abs(cx - border // 2) + np.abs(cy - border // 2)) < (border // 3)
    corner[diamond_c] = [210, 175, 55]

    arr[0:border, 0:border] = corner
    arr[0:border, width - border:] = corner[:, ::-1]
    arr[height - border:, 0:border] = corner[::-1, :]
    arr[height - border:, width - border:] = corner[::-1, ::-1]

    # --- Add noise to borders ---
    print("  Adding texture...")
    border_noise = rng.integers(-8, 9, (height, width), dtype=np.int8)
    # Only apply to border region
    for region in [
        (slice(0, border), slice(0, width)),
        (slice(height - border, height), slice(0, width)),
        (slice(border, height - border), slice(0, border)),
        (slice(border, height - border), slice(width - border, width)),
    ]:
        for c in range(3):
            arr[region[0], region[1], c] = np.clip(
                arr[region[0], region[1], c].astype(np.int16) + border_noise[region[0], region[1]],
                0, 255
            ).astype(np.uint8)

    # --- Save ---
    print(f"  Converting to image...")
    img = pyvips.Image.new_from_memory(arr.tobytes(), width, height, 3, "uchar")

    print(f"  Saving to {output_path} (uncompressed TIFF)...")
    img.tiffsave(output_path, compression="none", tile=False)

    del arr
    size_gb = Path(output_path).stat().st_size / (1024 ** 3)
    print(f"Done! {output_path}: {width}x{height} px, {size_gb:.2f} GB")
    print(f"Border: {border} px on all sides")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a demo rug image")
    parser.add_argument("--output", default="demo_rug.tiff", help="Output file path")
    args = parser.parse_args()
    generate_demo_rug(args.output)
