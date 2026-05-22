import base64
import io
import json
import logging
import os
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, request, send_from_directory

from image_processor import ImageProcessor, get_ocr

app = Flask(__name__, static_folder="static", static_url_path="")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
VALUE_PATH = os.path.join(DATA_DIR, "value.txt")

DEFAULT_SETTINGS = {
    "imageUrl": "",
    "maxThreshold": 0.2,
}

os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_value():
    if os.path.exists(VALUE_PATH):
        with open(VALUE_PATH, "r") as f:
            return float(f.read().strip())
    return None


def _save_value(value):
    with open(VALUE_PATH, "w") as f:
        f.write(str(value))


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    if not os.path.exists(CONFIG_PATH):
        return jsonify({}), 200
    return jsonify(_load_json(CONFIG_PATH, {}))


@app.put("/api/config")
def put_config():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    _save_json(CONFIG_PATH, data)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    settings = _load_json(SETTINGS_PATH, dict(DEFAULT_SETTINGS))
    return jsonify(settings)


@app.put("/api/settings")
def put_settings():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    settings = _load_json(SETTINGS_PATH, dict(DEFAULT_SETTINGS))
    settings.update(data)
    _save_json(SETTINGS_PATH, settings)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Value endpoint
# ---------------------------------------------------------------------------

@app.get("/api/value")
def get_value():
    value = _load_value()
    return jsonify({"value": value})


# ---------------------------------------------------------------------------
# Read endpoint
# ---------------------------------------------------------------------------

@app.post("/api/read")
def post_read():
    body = request.get_json(force=True, silent=True) or {}
    settings = _load_json(SETTINGS_PATH, dict(DEFAULT_SETTINGS))

    image_url = body.get("imageUrl") or settings.get("imageUrl", "")
    if not image_url:
        return jsonify({"error": "No imageUrl configured"}), 400

    if not os.path.exists(CONFIG_PATH):
        return jsonify({"error": "No config.json found. Please configure the processor first."}), 400

    config = _load_json(CONFIG_PATH, {})

    # Merge maxThreshold from settings into config sanity section
    max_threshold = settings.get("maxThreshold")
    if max_threshold is not None:
        config.setdefault("sanity", {})["maxThreshold"] = max_threshold

    # Validate URL scheme to prevent SSRF (only http/https allowed)
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        return jsonify({"error": "imageUrl must use http or https"}), 400

    # Fetch image
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to fetch image from %s: %s", image_url, exc)
        return jsonify({"error": "Failed to fetch image from the configured URL"}), 502

    image_bytes = resp.content

    previous = _load_value()

    # Save debug image to a temp buffer
    debug_path = os.path.join(DATA_DIR, "_debug.jpg")

    try:
        ip = ImageProcessor(image_bytes, config)
        result = ip.process(previous, debug=debug_path)
    except Exception as exc:
        logger.exception("Processing failed")
        return jsonify({"error": "Image processing failed"}), 500

    if result is None:
        return jsonify({"error": "Could not parse image"}), 422

    # Sanity checks
    if previous is not None:
        if result < previous:
            return jsonify({
                "error": f"Result {result} is less than previous {previous}",
                "value": result,
            }), 422
        if max_threshold is not None and result > previous + max_threshold:
            return jsonify({
                "error": f"Result {result} exceeds previous + {max_threshold}",
                "value": result,
            }), 422

    _save_value(result)

    # Read debug image as base64
    debug_image_b64 = None
    if os.path.exists(debug_path):
        with open(debug_path, "rb") as f:
            debug_image_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    return jsonify({
        "value": result,
        "previous": previous,
        "debugImage": debug_image_b64,
    })


if __name__ == "__main__":
    # Pre-load OCR model at startup so the first request isn't slow
    get_ocr()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
