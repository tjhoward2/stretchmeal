# Rug Design Pipeline

Automates storage optimization and on-demand scaling for rug design files. Handles the full workflow: PSD flattening, border detection, 9-slice scaling (borders stay crisp at any rug size), layered PSD scaling, and print-ready file generation.

## The Problem

Our designers' PSD files average 3.2 GB (some over 1 TB) due to hundreds of layers and embedded photographic textures. They've crashed the Design Server. Meanwhile, the actual print-ready image for our largest rug (10x14 ft at 150 DPI) is only 2-4 GB flattened. We need a way to flatten these files automatically, store them efficiently, and scale designs to any rug size without distorting decorative borders.

## What It Does

1. **Ingest** — Flattens a PSD to a master TIFF (strips all layers, keeps the composite), detects border zones, validates quality, archives the original
2. **Scale (flattened)** — Generates a print-ready TIFF for any rug size using 9-slice scaling (corners untouched, borders scaled in one axis, interior scaled freely)
3. **Scale (layered)** — Scales a layered PSD while preserving layer structure, groups, blend modes, and opacity — designers can still edit the output in Photoshop
4. **AI Upscale** — Uses Real-ESRGAN when scaling larger than the master, falls back to Lanczos gracefully
5. **S3 Storage** — Masters in S3 Standard, originals auto-archive to Glacier Deep Archive, scaled files cached on local disk
6. **Cache** — Cached print files auto-expire after 30 days
7. **Zone Tagging UI** — Web interface for viewing all designs, correcting auto-detected borders, and deleting designs

## Quick Start

**Prerequisites:** Python 3.10+, [libvips](https://www.libvips.org/) (`brew install vips` on macOS)

```bash
git clone https://github.com/tjhoward2/Design-Scaling-Pipeline.git
cd Design-Scaling-Pipeline/rug-pipeline
pip install -r requirements.txt
```

### Generate a test rug (skip if you have a real PSD)

```bash
python tools/generate_demo_rug.py --output demo_rug.tiff
```

### Run the pipeline

```bash
# Ingest a design
python pipeline.py ingest --psd demo_rug.tiff --design-id 001

# Set border zones (if auto-detection needs correction)
python pipeline.py set-zones --design-id 001 --top 800 --bottom 800 --left 800 --right 800

# Scale to print-ready TIFF (flattened)
python pipeline.py scale --design-id 001 --width 5 --height 7
python pipeline.py scale --design-id 001 --width 10 --height 14

# Scale a layered PSD (preserves layers for Photoshop editing)
python pipeline.py scale-psd --psd design.psd --zone-map zone_map.json --width 5 --height 7

# View cached outputs
open pipeline_data/cache/001/

# List all designs
python pipeline.py list

# Launch the web UI
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
# Ingest
python pipeline.py ingest --psd FILE --design-id ID

# Scale to flattened TIFF (print-ready)
python pipeline.py scale --design-id ID --width W --height H [--dpi 150]

# Scale to layered PSD (editable)
python pipeline.py scale-psd --design-id ID --width W --height H [--dpi 150]
python pipeline.py scale-psd --psd FILE --zone-map JSON --width W --height H

# Border management
python pipeline.py set-zones --design-id ID --top T --bottom B --left L --right R

# Maintenance
python pipeline.py cleanup-cache [--max-age-days 30]
python pipeline.py list
python pipeline.py validate --design-id ID

# Storage mode (default: local filesystem)
python pipeline.py --storage s3 ingest --psd FILE --design-id ID
```

## Storage Architecture

| Tier | Local Mode | S3 Mode | Purpose |
|------|-----------|---------|---------|
| **Masters** | `pipeline_data/masters/` | S3 Standard bucket | Flattened TIFFs, thumbnails, zone maps |
| **Archive** | `pipeline_data/archive/` | S3 → Glacier Deep Archive | Original PSDs (~$0.003/mo per 3 GB) |
| **Cache** | `pipeline_data/cache/` | Local disk | Scaled print files (30-day TTL) |

Switch between modes with `--storage local` or `--storage s3`.

## Layered PSD Scaling

The `scale-psd` command scales a PSD while preserving its layer tree:

- Each layer is classified by position relative to the border zones
- **Corner layers** — kept at original size, repositioned
- **Border layers** — scaled in one axis only (parallel to the border)
- **Interior layers** — uniformly scaled to fill the new interior space
- **Full-span layers** (backgrounds) — 9-slice scaled to target dimensions
- **Groups** — preserved recursively with all children scaled independently

**Supported:** Pixel layers, groups, blend modes, opacity

**Rasterized with warning:** Text layers, smart objects, adjustment layers, layer effects (visual appearance preserved, editability lost)

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

## Run Tests

```bash
pip install pytest "moto[s3]"
python -m pytest tests/ -v
```

75 tests covering: 9-slice engine, border detection, ingest/validation, resize/cache, storage backends (local + S3), and layered PSD scaling.

## Tech Stack

- **pyvips** — streams large images (1 TB+) without loading into RAM
- **OpenCV** — Canny edge detection for automatic border finding
- **psd-tools** — reads/writes layered PSD files
- **Real-ESRGAN** — AI upscaling via `realesrgan-ncnn-vulkan` binary
- **Flask** — zone tagging web UI
- **boto3** — S3 storage with Glacier lifecycle
