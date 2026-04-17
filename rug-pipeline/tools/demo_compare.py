#!/usr/bin/env python3
"""Generate side-by-side comparison images for demo purposes.

Creates a visual comparison showing:
1. The original master at full size
2. Scaled outputs at different rug sizes
3. A zoomed crop of the border region showing preservation

Usage:
    python tools/demo_compare.py --design-id demo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyvips

from config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Generate demo comparison images")
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--data-dir", default="./pipeline_data")
    parser.add_argument("--output-dir", default="./demo_output")
    args = parser.parse_args()

    config = PipelineConfig(base_dir=Path(args.data_dir))
    design_dir = config.masters_dir / args.design_id
    cache_dir = config.cache_dir / args.design_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load master
    master_path = design_dir / "master.tiff"
    if not master_path.exists():
        print(f"Master not found: {master_path}")
        sys.exit(1)

    master = pyvips.Image.new_from_file(str(master_path))
    zone_map = json.loads((design_dir / "zone_map.json").read_text())

    print(f"Master: {master.width}x{master.height} px")
    print(f"Borders: T={zone_map['border_top_px']} B={zone_map['border_bottom_px']} "
          f"L={zone_map['border_left_px']} R={zone_map['border_right_px']}")

    # Save master preview (resized to ~1500px wide for easy viewing)
    preview_scale = 1500 / master.width
    master_preview = master.resize(preview_scale, kernel="lanczos3")
    master_preview.pngsave(str(output_dir / "01_master_original.png"))
    print(f"Saved: 01_master_original.png ({master_preview.width}x{master_preview.height})")

    # Save corner crop from master (top-left, showing border detail)
    bt = zone_map["border_top_px"]
    bl = zone_map["border_left_px"]
    crop_size = max(bt, bl) + 200  # border + some interior
    crop_size = min(crop_size, master.width, master.height)
    corner_crop = master.crop(0, 0, crop_size, crop_size)
    # Scale up for visibility
    if corner_crop.width < 800:
        corner_crop = corner_crop.resize(800 / corner_crop.width, kernel="lanczos3")
    corner_crop.pngsave(str(output_dir / "02_border_detail.png"))
    print(f"Saved: 02_border_detail.png (top-left corner crop)")

    # Save previews of each cached scaled version
    if cache_dir.exists():
        idx = 3
        for tiff in sorted(cache_dir.glob("*.tiff")):
            scaled = pyvips.Image.new_from_file(str(tiff))
            # Resize to ~1500px wide for preview
            s = 1500 / scaled.width
            preview = scaled.resize(s, kernel="lanczos3")
            name = f"{idx:02d}_scaled_{tiff.stem}.png"
            preview.pngsave(str(output_dir / name))
            print(f"Saved: {name} ({scaled.width}x{scaled.height} → {preview.width}x{preview.height} preview)")

            # Also save corner crop from scaled version for comparison
            scaled_crop_size = min(crop_size, scaled.width, scaled.height)
            scaled_corner = scaled.crop(0, 0, scaled_crop_size, scaled_crop_size)
            if scaled_corner.width < 800:
                scaled_corner = scaled_corner.resize(800 / scaled_corner.width, kernel="lanczos3")
            crop_name = f"{idx:02d}_scaled_{tiff.stem}_border_detail.png"
            scaled_corner.pngsave(str(output_dir / crop_name))
            print(f"Saved: {crop_name} (border detail from scaled)")
            idx += 1

    print(f"\nAll comparison images saved to: {output_dir}/")
    print("Open them with: open demo_output/")


if __name__ == "__main__":
    main()
