# Rug Design Pipeline

Automates storage optimization and on-demand scaling for rug design files. Handles the full workflow: PSD flattening, border detection, 9-slice scaling (borders stay crisp at any rug size), and print-ready file generation.

## The Problem

Our designers' PSD files average 3.2 GB (some over 1 TB) due to hundreds of layers and embedded photographic textures. They've crashed the Design Server. Meanwhile, the actual print-ready image for our largest rug (10x14 ft at 150 DPI) is only 2-4 GB flattened. We need a way to flatten these files automatically, store them efficiently, and scale designs to any rug size without distorting decorative borders.

## What It Does

1. **Ingest** — Flattens a PSD to a master TIFF (strips all layers, keeps the composite), detects border zones, validates quality, archives the original
2. **Scale** — Generates a print-ready file for any rug size using 9-slice scaling (corners untouched, borders scaled in one axis, interior scaled freely)
3. **AI Upscale** — Uses Real-ESRGAN when scaling larger than the master, falls back to Lanczos gracefully
4. **Cache** — Cached print files auto-expire after 30 days
5. **Zone Tagging UI** — Web interface for correcting auto-detected borders when confidence is low

## Quick Start

**Prerequisites:** Python 3.10+, [libvips](https://www.libvips.org/) (`brew install vips` on macOS)

```bash
git clone https://github.com/tjhoward2/stretchmeal.git
cd stretchmeal/rug-pipeline
pip install -r requirements.txt

# Generate a test rug image (skip if you have a real PSD)
python -c "
import numpy as np, pyvips
w, h, b = 3600, 5040, 400
arr = np.zeros((h, w, 3), dtype=np.uint8)
arr[b:h-b, b:w-b] = [40, 60, 160]
arr[:b, :] = [150, 40, 40]; arr[h-b:, :] = [150, 40, 40]
arr[:, :b] = [150, 40, 40]; arr[:, w-b:] = [150, 40, 40]
for cy in [range(b), range(h-b, h)]:
    for cx in [range(b), range(w-b, w)]:
        for y in cy:
            for x in cx: arr[y, x] = [200, 170, 50]
pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, 'uchar').tiffsave('test_rug.tiff', compression='lzw')
print('Created test_rug.tiff')
"

# Ingest the design
python pipeline.py ingest --psd test_rug.tiff --design-id 001

# Scale to different rug sizes
python pipeline.py scale --design-id 001 --width 5 --height 7
python pipeline.py scale --design-id 001 --width 8 --height 10
python pipeline.py scale --design-id 001 --width 10 --height 14

# See what you have
python pipeline.py list

# Launch the zone tagging UI
python -m web.app --port 5000
# Open http://localhost:5000
```

## Test with a Real PSD

The smoke test runs the full pipeline against a real file and reports diagnostics without modifying the original:

```bash
python tests/smoke_test_real_psd.py /path/to/design.psd
```

## CLI Reference

```bash
python pipeline.py ingest --psd FILE --design-id ID      # Flatten + detect + validate + archive
python pipeline.py scale --design-id ID --width W --height H [--dpi 150]
python pipeline.py set-zones --design-id ID --top T --bottom B --left L --right R
python pipeline.py cleanup-cache [--max-age-days 30]
python pipeline.py list
python pipeline.py validate --design-id ID
```

## Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## How 9-Slice Scaling Works

Traditional scaling distorts decorative borders. 9-slice splits the image into 9 zones:

```
┌────────┬──────────────────┬────────┐
│ Corner  │   Top Border     │ Corner  │  ← Fixed height
├────────┼──────────────────┼────────┤
│  Left  │                  │ Right  │
│ Border │    Interior      │ Border │  ← Scales freely
│        │                  │        │
├────────┼──────────────────┼────────┤
│ Corner  │  Bottom Border   │ Corner  │  ← Fixed height
└────────┴──────────────────┴────────┘
   Fixed      Scales width      Fixed
   width                        width
```

- **Corners**: Never resized — pixel-perfect at any rug size
- **Border strips**: Scaled in one axis only (top/bottom stretch width, left/right stretch height)
- **Interior**: Scaled freely to fill the remaining space
