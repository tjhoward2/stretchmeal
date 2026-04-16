"""Flask web app for zone tagging UI.

Provides a minimal interface for reviewing and correcting auto-detected
border zones on rug designs. Works with both local and S3 storage backends.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig
from storage.backend import StorageBackend, create_backend

logger = logging.getLogger(__name__)

app = Flask(__name__)
config = PipelineConfig()
storage: StorageBackend = create_backend("local")
masters_bucket: str = str(config.masters_dir)


def _get_design_ids() -> list[str]:
    """Get all design IDs from the storage backend."""
    all_keys = storage.list_keys("", masters_bucket)
    return sorted({k.split("/")[0] for k in all_keys if "/" in k})


@app.route("/")
def index():
    """Serve the zone tagging UI."""
    return render_template("index.html")


@app.route("/api/designs")
def list_designs():
    """List designs, optionally filtered by needs_review status."""
    needs_review = request.args.get("needs_review", "").lower() == "true"

    designs = []
    for design_id in _get_design_ids():
        zone_key = f"{design_id}/zone_map.json"
        if not storage.exists(zone_key, masters_bucket):
            continue

        zone_data = storage.read_json(zone_key, masters_bucket)

        if needs_review and not zone_data.get("needs_review", False):
            continue

        designs.append({
            "design_id": design_id,
            "has_thumbnail": storage.exists(f"{design_id}/thumbnail.png", masters_bucket),
            "zone_map": zone_data,
        })

    return jsonify(designs)


@app.route("/api/designs/<design_id>/thumbnail")
def get_thumbnail(design_id: str):
    """Serve a design's thumbnail image."""
    thumb_key = f"{design_id}/thumbnail.png"
    if not storage.exists(thumb_key, masters_bucket):
        return jsonify({"error": "Thumbnail not found"}), 404

    # Download to temp file and serve
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        local_path = Path(tmp.name)

    storage.get(thumb_key, masters_bucket, local_path)
    return send_file(local_path, mimetype="image/png")


@app.route("/api/designs/<design_id>/zones", methods=["GET"])
def get_zones(design_id: str):
    """Get the current zone map for a design."""
    zone_key = f"{design_id}/zone_map.json"
    if not storage.exists(zone_key, masters_bucket):
        return jsonify({"error": "Zone map not found"}), 404
    return jsonify(storage.read_json(zone_key, masters_bucket))


@app.route("/api/designs/<design_id>/zones", methods=["POST"])
def update_zones(design_id: str):
    """Update a design's zone map with manually corrected borders."""
    zone_key = f"{design_id}/zone_map.json"
    if not storage.exists(zone_key, masters_bucket):
        return jsonify({"error": "Zone map not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Load existing zone map and update border values
    zone_map = storage.read_json(zone_key, masters_bucket)

    for key in ("border_top_px", "border_bottom_px", "border_left_px", "border_right_px"):
        if key in data:
            zone_map[key] = int(data[key])

    zone_map["detection_method"] = "manual"
    zone_map["confidence"] = 1.0
    zone_map["needs_review"] = False

    storage.write_json(zone_map, zone_key, masters_bucket)
    logger.info("Updated zone map for design %s: %s", design_id, zone_map)

    return jsonify(zone_map)


def create_app(data_dir: str | None = None, storage_backend: str | None = None) -> Flask:
    """Factory function for creating the Flask app with custom config."""
    global config, storage, masters_bucket
    if data_dir:
        config = PipelineConfig(base_dir=Path(data_dir))
    if storage_backend:
        config.storage_backend = storage_backend

    storage = create_backend(
        config.storage_backend,
        region=config.s3_region,
        glacier_transition_days=config.s3_glacier_transition_days,
    )
    if config.storage_backend == "s3":
        masters_bucket = config.s3_masters_bucket
    else:
        masters_bucket = str(config.masters_dir)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zone tagging web UI")
    parser.add_argument("--data-dir", type=str, default="./pipeline_data")
    parser.add_argument("--storage", choices=["local", "s3"], default="local")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    create_app(data_dir=args.data_dir, storage_backend=args.storage)
    app.run(port=args.port, debug=args.debug)
