import base64
import copy
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, request, send_from_directory
try:
    import paho.mqtt.publish as mqtt_publish
except ImportError:  # pragma: no cover - optional dependency at runtime
    mqtt_publish = None

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
    "paddleOcrLang": "en",
    "darkMode": False,
    "autoReadIntervalSec": 0,
    "mqtt": {
        "enabled": False,
        "host": "",
        "port": 1883,
        "topic": "watermeter/value",
        "username": "",
        "password": "",
        "qos": 0,
        "retain": False,
        "clientId": "",
    },
}


def _normalize_ocr_lang(value):
    normalized = str(value or "en").strip().lower()
    if normalized in ("standard", "english", "english_g2"):
        return "en"
    return normalized


def _settings_ocr_lang(settings: dict):
    if not isinstance(settings, dict):
        return "en"
    if "paddleOcrLang" in settings:
        return _normalize_ocr_lang(settings.get("paddleOcrLang"))
    return _normalize_ocr_lang(settings.get("easyOcrModel"))

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


def _load_settings():
    loaded = _load_json(SETTINGS_PATH, {})
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if isinstance(loaded, dict):
        settings.update({k: v for k, v in loaded.items() if k != "mqtt"})
        settings["paddleOcrLang"] = _settings_ocr_lang(loaded)
        mqtt_defaults = copy.deepcopy(DEFAULT_SETTINGS["mqtt"])
        loaded_mqtt = loaded.get("mqtt") if isinstance(loaded.get("mqtt"), dict) else {}
        mqtt_defaults.update(loaded_mqtt)
        settings["mqtt"] = mqtt_defaults
    return settings


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_settings(data):
    settings = _load_settings()
    if not isinstance(data, dict):
        return settings

    if "imageUrl" in data:
        settings["imageUrl"] = str(data.get("imageUrl") or "").strip()
    if "maxThreshold" in data:
        settings["maxThreshold"] = _to_float(data.get("maxThreshold"), DEFAULT_MAX_THRESHOLD)
    if "easyOcrModel" in data:
        settings["paddleOcrLang"] = _normalize_ocr_lang(data.get("easyOcrModel"))
    if "paddleOcrLang" in data:
        settings["paddleOcrLang"] = _normalize_ocr_lang(data.get("paddleOcrLang"))
    if "darkMode" in data:
        settings["darkMode"] = _as_bool(data.get("darkMode"))
    if "autoReadIntervalSec" in data:
        settings["autoReadIntervalSec"] = max(0, _to_int(data.get("autoReadIntervalSec"), 0))

    mqtt_data = data.get("mqtt")
    if isinstance(mqtt_data, dict):
        mqtt_settings = settings["mqtt"]
        if "enabled" in mqtt_data:
            mqtt_settings["enabled"] = _as_bool(mqtt_data.get("enabled"))
        if "host" in mqtt_data:
            mqtt_settings["host"] = str(mqtt_data.get("host") or "").strip()
        if "port" in mqtt_data:
            mqtt_settings["port"] = max(1, _to_int(mqtt_data.get("port"), 1883))
        if "topic" in mqtt_data:
            mqtt_settings["topic"] = str(mqtt_data.get("topic") or "").strip()
        if "username" in mqtt_data:
            mqtt_settings["username"] = str(mqtt_data.get("username") or "")
        if "password" in mqtt_data:
            mqtt_settings["password"] = str(mqtt_data.get("password") or "")
        if "qos" in mqtt_data:
            mqtt_settings["qos"] = min(2, max(0, _to_int(mqtt_data.get("qos"), 0)))
        if "retain" in mqtt_data:
            mqtt_settings["retain"] = _as_bool(mqtt_data.get("retain"))
        if "clientId" in mqtt_data:
            mqtt_settings["clientId"] = str(mqtt_data.get("clientId") or "").strip()

    return settings


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


def _publish_mqtt_value(value, settings):
    mqtt_settings = settings.get("mqtt") if isinstance(settings, dict) else None
    if not isinstance(mqtt_settings, dict) or not mqtt_settings.get("enabled"):
        return
    if mqtt_publish is None:
        logger.warning("MQTT enabled but paho-mqtt is unavailable")
        return
    host = str(mqtt_settings.get("host") or "").strip()
    topic = str(mqtt_settings.get("topic") or "").strip()
    if not host or not topic:
        logger.warning("MQTT enabled but host/topic are not configured")
        return
    username = mqtt_settings.get("username") or None
    password = mqtt_settings.get("password") or None
    auth = {"username": username, "password": password} if username else None
    try:
        mqtt_publish.single(
            topic=topic,
            payload=str(value),
            hostname=host,
            port=max(1, _to_int(mqtt_settings.get("port"), 1883)),
            qos=min(2, max(0, _to_int(mqtt_settings.get("qos"), 0))),
            retain=_as_bool(mqtt_settings.get("retain")),
            auth=auth,
            client_id=str(mqtt_settings.get("clientId") or ""),
            keepalive=10,
        )
    except Exception:
        logger.exception("Failed to publish MQTT value")


def _build_processing_config(config, settings):
    merged = copy.deepcopy(config)
    max_threshold = settings.get("maxThreshold")
    if max_threshold is not None:
        merged.setdefault("sanity", {})["maxThreshold"] = max_threshold
    return merged


def _run_processing(image_url, config, previous, settings, debug_path):
    image_bytes, fetch_error, status_code = _fetch_image_bytes(image_url)
    if fetch_error:
        return None, fetch_error, status_code
    try:
        ip = ImageProcessor(image_bytes, config, ocr_model=_settings_ocr_lang(settings))
        details = ip.process_with_details(previous, debug=debug_path)
        return details, None, None
    except ValueError:
        logger.exception("Processing failed")
        return None, "Image processing failed: check pointer color/threshold/decolor settings", 500
    except Exception:
        logger.exception("Processing failed with unexpected error")
        return None, "Image processing failed — check server logs for details", 500


def _run_read_cycle(
    image_url_override=None,
    config_override=None,
    persist_value=False,
    include_test_details=False,
    debug_filename="_debug.jpg",
):
    settings = _load_settings()
    image_url = image_url_override or settings.get("imageUrl", "")
    if not image_url:
        return None, "No imageUrl configured", 400

    config = config_override if config_override is not None else _load_json(CONFIG_PATH, dict(DEFAULT_CONFIG))
    config = _build_processing_config(config, settings)
    previous = _load_value()
    debug_path = os.path.join(DATA_DIR, debug_filename)

    details, err, status = _run_processing(image_url, config, previous, settings, debug_path)
    if err:
        return {"error": err, "debugImage": _debug_image_data_url(debug_path)}, err, status

    result = details["value"]
    max_threshold = settings.get("maxThreshold")

    if persist_value and previous is not None:
        if result < previous:
            return {
                "error": f"Result {result} is less than previous {previous}",
                "value": result,
                "debugImage": _debug_image_data_url(debug_path),
            }, "sanity", 422
        if max_threshold is not None and result > previous + max_threshold:
            return {
                "error": f"Result {result} exceeds previous + {max_threshold}",
                "value": result,
                "debugImage": _debug_image_data_url(debug_path),
            }, "sanity", 422

    if persist_value:
        _save_value(result)
        _publish_mqtt_value(result, settings)

    payload = {
        "value": result,
        "previous": previous,
        "debugImage": _debug_image_data_url(debug_path),
    }
    if include_test_details:
        payload.update({
            "digits": details["digits"],
            "decimalDigits": details["decimal_digits"],
            "analogDigits": details["analog_digits"],
            "decimalUsed": details["decimal_used"],
            "digitDetails": details["digit_details"],
            "decimalDigitDetails": details["decimal_digit_details"],
            "analogDetails": details["analog_details"],
            "decimalSource": details["decimal_source"],
        })

    return payload, None, None


_auto_read_lock = threading.Lock()
_auto_read_last_run = 0.0
_auto_read_started = False


def _auto_reader_loop():
    global _auto_read_last_run
    while True:
        try:
            settings = _load_settings()
            interval = max(0, _to_int(settings.get("autoReadIntervalSec"), 0))
            if interval <= 0:
                time.sleep(1)
                continue
            now = time.time()
            if now - _auto_read_last_run < interval:
                time.sleep(1)
                continue
            if not _auto_read_lock.acquire(blocking=False):
                time.sleep(1)
                continue
            try:
                payload, err, _ = _run_read_cycle(
                    persist_value=True,
                    include_test_details=False,
                    debug_filename="_debug_auto.jpg",
                )
                if err:
                    logger.warning("Auto-read failed: %s", payload.get("error"))
                _auto_read_last_run = time.time()
            finally:
                _auto_read_lock.release()
        except Exception:
            logger.exception("Auto-read loop failed")
            time.sleep(2)
        time.sleep(1)


def _ensure_auto_reader_started():
    global _auto_read_started
    if _auto_read_started:
        return
    threading.Thread(target=_auto_reader_loop, daemon=True).start()
    _auto_read_started = True


@app.before_request
def _ensure_auto_reader_from_settings():
    if _to_int(_load_settings().get("autoReadIntervalSec"), 0) > 0:
        _ensure_auto_reader_started()


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
    return jsonify(_load_settings())


@app.put("/api/settings")
def put_settings():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    settings = _normalize_settings(data)
    _save_json(SETTINGS_PATH, settings)
    if _to_int(settings.get("autoReadIntervalSec"), 0) > 0:
        _ensure_auto_reader_started()
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
    with _auto_read_lock:
        payload, err, status = _run_read_cycle(
            image_url_override=body.get("imageUrl"),
            persist_value=True,
            include_test_details=False,
            debug_filename="_debug.jpg",
        )
    if err:
        return jsonify(payload), status
    return jsonify(payload)


@app.post("/api/read/test")
def post_read_test():
    body = request.get_json(force=True, silent=True) or {}

    config = body.get("config")
    if config is None:
        config = _load_json(CONFIG_PATH, dict(DEFAULT_CONFIG))
    elif not isinstance(config, dict):
        return jsonify({"error": "config must be a JSON object"}), 400

    with _auto_read_lock:
        payload, err, status = _run_read_cycle(
            image_url_override=body.get("imageUrl"),
            config_override=config,
            persist_value=False,
            include_test_details=True,
            debug_filename="_debug_test.jpg",
        )
    if err:
        return jsonify(payload), status
    return jsonify(payload)


if __name__ == "__main__":
    # Pre-load OCR model at startup so the first request isn't slow
    get_ocr(_settings_ocr_lang(_load_settings()))
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
