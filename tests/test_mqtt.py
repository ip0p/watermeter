import importlib
import json
import sys
import types
import unittest
from unittest import mock


def load_web_server():
    fake_image_processor = types.ModuleType("image_processor")
    fake_image_processor.ImageProcessor = object
    fake_image_processor.get_ocr = lambda *args, **kwargs: None
    with mock.patch.dict(sys.modules, {"image_processor": fake_image_processor}):
        sys.modules.pop("web_server", None)
        return importlib.import_module("web_server")


class FakePublishModule:
    def __init__(self):
        self.calls = []

    def multiple(self, **kwargs):
        self.calls.append(kwargs)


class TestMqttPublishing(unittest.TestCase):
    def setUp(self):
        self.web_server = load_web_server()
        self.web_server._mqtt_last_discovery_signature = None
        self.fake_publish = FakePublishModule()
        self.web_server.mqtt_publish = self.fake_publish

    def test_publish_includes_home_assistant_discovery_once(self):
        settings = {
            "mqtt": {
                "enabled": True,
                "host": "mqtt.local",
                "port": 1883,
                "topic": "watermeter/value",
                "discoveryEnabled": True,
                "discoveryPrefix": "homeassistant",
                "username": "",
                "password": "",
                "qos": 1,
                "retain": True,
                "clientId": "meter-1",
            }
        }

        self.web_server._publish_mqtt_value(123.456, settings)
        self.web_server._publish_mqtt_value(123.457, settings)

        self.assertEqual(len(self.fake_publish.calls), 2)

        first_messages = self.fake_publish.calls[0]["msgs"]
        self.assertEqual(len(first_messages), 3)
        self.assertEqual(first_messages[0]["topic"], "homeassistant/sensor/meter_1/config")
        discovery_payload = json.loads(first_messages[0]["payload"])
        self.assertEqual(discovery_payload["state_topic"], "watermeter/value")
        self.assertEqual(discovery_payload["availability_topic"], "watermeter/value/availability")
        self.assertEqual(discovery_payload["device"]["name"], "meter-1")
        self.assertEqual(first_messages[1]["topic"], "watermeter/value/availability")
        self.assertEqual(first_messages[2]["topic"], "watermeter/value")
        self.assertEqual(first_messages[2]["payload"], "123.456")

        second_messages = self.fake_publish.calls[1]["msgs"]
        self.assertEqual(len(second_messages), 2)
        self.assertEqual(second_messages[0]["topic"], "watermeter/value/availability")
        self.assertEqual(second_messages[1]["topic"], "watermeter/value")
        self.assertEqual(second_messages[1]["payload"], "123.457")

    def test_settings_normalization_applies_discovery_defaults(self):
        normalized = self.web_server._normalize_settings(
            {
                "mqtt": {
                    "enabled": True,
                    "host": "mqtt.local",
                    "topic": "watermeter/value",
                    "discoveryEnabled": False,
                    "discoveryPrefix": "",
                }
            }
        )

        self.assertFalse(normalized["mqtt"]["discoveryEnabled"])
        self.assertEqual(normalized["mqtt"]["discoveryPrefix"], "homeassistant")
