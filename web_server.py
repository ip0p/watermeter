import base64
import ipaddress
import io
import json
import logging
import os
import socket
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, request, send_from_directory

from image_processor import ImageProcessor, get_ocr

app = Flask(__name__, static_folder="static", static_url_path="")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ALLOW_PRIVATE_URLS = os.environ.get("ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
VALUE_PATH = os.path.join(DATA_DIR, "value.txt")

DEFAULT_MAX_THRESHOLD = 0.2

DEFAULT_SETTINGS = {
    "imageUrl": "",
    "maxThreshold": DEFAULT_MAX_THRESHOLD,
}

DEFAULT_CONFIG = {
    "sanity": {
        "maxThreshold": DEFAULT_MAX_THRESHOLD
    },
    "image": {
        "rotate": 0,
        "crop": {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0
        }
    },
    "digits": [],
    "decimal_digits": [],
    "decimal_analogs": [],
    "postprocessing": {
        "digits": {
            "brightness": 0,
            "contrast": 0,
            "decolor": False
        },
        "analog": {
            "brightness": 0,
            "contrast": 0,
            "binaryThreshold": 128,
            "decolor": False
        }
    }
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


def _validate_image_url(url: str):
    """
    Validate the URL and return (resolved_ip: str, error: str | None).
    Blocks non-http(s) schemes and private/reserved/multicast IPs to prevent SSRF.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None, "imageUrl must use http or https"
    hostname = parsed.hostname
    if not hostname:
        return None, "imageUrl has no hostname"
    try:
        resolved = socket.gethostbyname(hostname)
        addr = ipaddress.ip_address(resolved)
    except (socket.gaierror, ValueError):
        return None, "Could not resolve imageUrl hostname"
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        if not ALLOW_PRIVATE_URLS:
            return None, "imageUrl resolves to a private, reserved, or multicast address"
    return resolved, None


def _build_safe_url(original_url: str, resolved_ip: str) -> tuple[str, dict]:
    """
    Rewrite the URL to use the pre-resolved IP to prevent DNS rebinding.
    Returns (rewritten_url, headers_with_host).
    """
    parsed = urlparse(original_url)
    host_header = parsed.netloc  # preserve original host (with port) for the Host header
    port = parsed.port
    netloc = resolved_ip if port is None else f"{resolved_ip}:{port}"
    rewritten = parsed._replace(netloc=netloc).geturl()
    return rewritten, {"Host": host_header}


def _fetch_image_bytes(image_url: str):
    resolved_ip, url_error = _validate_image_url(image_url)
    if url_error:
        return None, url_error, 400

    safe_url, extra_headers = _build_safe_url(image_url, resolved_ip)
    try:
        # URL is validated and rewritten to a resolved IP with private-network checks in _validate_image_url.
        resp = requests.get(safe_url, timeout=10, allow_redirects=False, headers=extra_headers)  # lgtm[py/full-ssrf]
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to fetch image from %s: %s", image_url, exc)
        return None, "Failed to fetch image from the configured URL", 502
    return resp.content, None, None


def _debug_image_data_url(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return None


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
    return jsonify(_load_json(CONFIG_PATH, dict(DEFAULT_CONFIG)))


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

    config = _load_json(CONFIG_PATH, dict(DEFAULT_CONFIG))

    # Merge maxThreshold from settings into config sanity section
    max_threshold = settings.get("maxThreshold")
    if max_threshold is not None:
        config.setdefault("sanity", {})["maxThreshold"] = max_threshold

    image_bytes, fetch_error, status_code = _fetch_image_bytes(image_url)
    if fetch_error:
        return jsonify({"error": fetch_error}), status_code

    previous = _load_value()

    # Save debug image to a temp buffer
    debug_path = os.path.join(DATA_DIR, "_debug.jpg")

    try:
        ip = ImageProcessor(image_bytes, config)
        result = ip.process(previous, debug=debug_path)
    except ValueError as exc:
        # ValueError messages are explicitly crafted to be user-facing (e.g. missing pointer color,
        # bad threshold), so it is safe and helpful to surface them directly.
        logger.exception("Processing failed")
        debug_image_b64 = _debug_image_data_url(debug_path)
        return jsonify({"error": f"Image processing failed: {exc}", "debugImage": debug_image_b64}), 500  # lgtm[py/stack-trace-exposure]
    except Exception as exc:
        # Unexpected errors: log full detail server-side, return a generic message to the client.
        logger.exception("Processing failed with unexpected error")
        debug_image_b64 = _debug_image_data_url(debug_path)
        return jsonify({"error": "Image processing failed — check server logs for details", "debugImage": debug_image_b64}), 500

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
    debug_image_b64 = _debug_image_data_url(debug_path)

    return jsonify({
        "value": result,
        "previous": previous,
        "debugImage": debug_image_b64,
    })


@app.post("/api/read/test")
def post_read_test():
    body = request.get_json(force=True, silent=True) or {}
    settings = _load_json(SETTINGS_PATH, dict(DEFAULT_SETTINGS))

    image_url = body.get("imageUrl") or settings.get("imageUrl", "")
    if not image_url:
        return jsonify({"error": "No imageUrl configured"}), 400

    config = body.get("config")
    if config is None:
        config = _load_json(CONFIG_PATH, dict(DEFAULT_CONFIG))
    elif not isinstance(config, dict):
        return jsonify({"error": "config must be a JSON object"}), 400

    image_bytes, fetch_error, status_code = _fetch_image_bytes(image_url)
    if fetch_error:
        return jsonify({"error": fetch_error}), status_code

    previous = _load_value()
    debug_path = os.path.join(DATA_DIR, "_debug_test.jpg")

    try:
        ip = ImageProcessor(image_bytes, config)
        details = ip.process_with_details(previous, debug=debug_path)
    except ValueError as exc:
        logger.exception("Test processing failed")
        debug_image_b64 = _debug_image_data_url(debug_path)
        return jsonify({"error": f"Image processing failed: {exc}", "debugImage": debug_image_b64}), 500  # lgtm[py/stack-trace-exposure]
    except Exception:
        logger.exception("Test processing failed with unexpected error")
        debug_image_b64 = _debug_image_data_url(debug_path)
        return jsonify({"error": "Image processing failed — check server logs for details", "debugImage": debug_image_b64}), 500

    debug_image_b64 = _debug_image_data_url(debug_path)
    return jsonify({
        "value": details["value"],
        "previous": previous,
        "digits": details["digits"],
        "decimalDigits": details["decimal_digits"],
        "digitDetails": details["digit_details"],
        "decimalDigitDetails": details["decimal_digit_details"],
        "analogDetails": details["analog_details"],
        "decimalSource": details["decimal_source"],
        "debugImage": debug_image_b64,
    })


if __name__ == "__main__":
    # Pre-load OCR model at startup so the first request isn't slow
    get_ocr()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
