import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


def load_web_server(data_dir):
    fake_image_processor = types.ModuleType("image_processor")
    fake_image_processor.ImageProcessor = object
    fake_image_processor.get_ocr = lambda *args, **kwargs: None
    with mock.patch.dict(sys.modules, {"image_processor": fake_image_processor}):
        with mock.patch.dict(os.environ, {"DATA_DIR": data_dir}):
            sys.modules.pop("web_server", None)
            return importlib.import_module("web_server")


class TestValueEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.web_server = load_web_server(self.tmp_dir)
        self.web_server.VALUE_PATH = os.path.join(self.tmp_dir, "value.txt")
        # Disable MQTT publishing
        self.web_server.mqtt_publish = None
        self.client = self.web_server.app.test_client()

    def tearDown(self):
        if os.path.exists(self.web_server.VALUE_PATH):
            os.remove(self.web_server.VALUE_PATH)

    def test_get_value_returns_null_when_no_file(self):
        r = self.client.get("/api/value")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.get_json()["value"])

    def test_get_value_returns_stored_value(self):
        with open(self.web_server.VALUE_PATH, "w") as f:
            f.write("123.456")
        r = self.client.get("/api/value")
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(r.get_json()["value"], 123.456)

    def test_put_value_sets_value(self):
        r = self.client.put(
            "/api/value",
            json={"value": 456.789},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(r.get_json()["value"], 456.789)
        self.assertTrue(os.path.exists(self.web_server.VALUE_PATH))
        with open(self.web_server.VALUE_PATH) as f:
            self.assertAlmostEqual(float(f.read()), 456.789)

    def test_put_value_rejects_missing_field(self):
        r = self.client.put("/api/value", json={}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_put_value_rejects_non_numeric(self):
        r = self.client.put(
            "/api/value",
            json={"value": "abc"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_put_value_rejects_negative(self):
        r = self.client.put(
            "/api/value",
            json={"value": -1.0},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_delete_value_removes_file(self):
        with open(self.web_server.VALUE_PATH, "w") as f:
            f.write("100.0")
        r = self.client.delete("/api/value")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(self.web_server.VALUE_PATH))

    def test_delete_value_is_idempotent(self):
        # Deleting when no file exists should succeed
        r = self.client.delete("/api/value")
        self.assertEqual(r.status_code, 200)

    def test_get_value_after_reset_returns_null(self):
        with open(self.web_server.VALUE_PATH, "w") as f:
            f.write("42.0")
        self.client.delete("/api/value")
        r = self.client.get("/api/value")
        self.assertIsNone(r.get_json()["value"])


if __name__ == "__main__":
    unittest.main()

