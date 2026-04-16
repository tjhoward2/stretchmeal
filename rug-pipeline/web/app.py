"""Flask web app for zone tagging UI.

Provides a minimal interface for reviewing and correcting auto-detected
border zones on rug designs.
"""

import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig

logger = logging.getLogger(__name__)

app = Flask(__name__)
config = PipelineConfig()


@app.route("/")
def index():
    """Serve the zone tagging UI."""
    return render_template("index.html")


@app.route("/api/designs")
def list_designs():
    """List designs, optionally filtered by needs_review status."""
    needs_review = request.args.get("needs_review", "").lower() == "true"
    masters_dir = config.masters_dir

    designs = []
    if masters_dir.exists():
        for design_dir in sorted(masters_dir.iterdir()):
            if not design_dir.is_dir():
                continue

            zone_path = design_dir / "zone_map.json"
            if not zone_path.exists():
                continue

            zone_data = json.loads(zone_path.read_text())

            if needs_review and not zone_data.get("needs_review", False):
                continue

            designs.append({
                "design_id": design_dir.name,
                "has_thumbnail": (design_dir / "thumbnail.png").exists(),
                "zone_map": zone_data,
            })

    return jsonify(designs)


@app.route("/api/designs/<design_id>/thumbnail")
def get_thumbnail(design_id: str):
    """Serve a design's thumbnail image."""
    thumb_path = config.masters_dir / design_id / "thumbnail.png"
    if not thumb_path.exists():
        return jsonify({"error": "Thumbnail not found"}), 404
    return send_file(thumb_path, mimetype="image/png")


@app.route("/api/designs/<design_id>/zones", methods=["GET"])
def get_zones(design_id: str):
    """Get the current zone map for a design."""
    zone_path = config.masters_dir / design_id / "zone_map.json"
    if not zone_path.exists():
        return jsonify({"error": "Zone map not found"}), 404
    return jsonify(json.loads(zone_path.read_text()))


@app.route("/api/designs/<design_id>/zones", methods=["POST"])
def update_zones(design_id: str):
    """Update a design's zone map with manually corrected borders."""
    zone_path = config.masters_dir / design_id / "zone_map.json"
    if not zone_path.exists():
        return jsonify({"error": "Zone map not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Load existing zone map and update border values
    zone_map = json.loads(zone_path.read_text())

    for key in ("border_top_px", "border_bottom_px", "border_left_px", "border_right_px"):
        if key in data:
            zone_map[key] = int(data[key])

    zone_map["detection_method"] = "manual"
    zone_map["confidence"] = 1.0
    zone_map["needs_review"] = False

    zone_path.write_text(json.dumps(zone_map, indent=2))
    logger.info("Updated zone map for design %s: %s", design_id, zone_map)

    return jsonify(zone_map)


def create_app(data_dir: str | None = None) -> Flask:
    """Factory function for creating the Flask app with custom config."""
    global config
    if data_dir:
        config = PipelineConfig(base_dir=Path(data_dir))
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zone tagging web UI")
    parser.add_argument("--data-dir", type=str, default="./pipeline_data")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = PipelineConfig(base_dir=Path(args.data_dir))
    app.run(port=args.port, debug=args.debug)
