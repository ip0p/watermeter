import unittest
from glob import glob
import os
from unittest import mock

import cv2
import numpy as np

from image_processor import ImageProcessor


class TestImagesInTests(unittest.TestCase):
    def test_all_images(self):
        images = glob("tests/*.png")
        images.sort()
        self.assertGreater(len(images), 0)
        previous_value = None
        last_valid_config = None
        for image in images:
            image_id = os.path.basename(image)[:-4]
            image_path = os.path.join("tests", image_id + ".png")
            config_path = os.path.join("tests", image_id + ".json")
            if os.path.exists(config_path):
                last_valid_config = config_path
            if last_valid_config is None:
                raise ValueError("No config found")
            previous_value_path = os.path.join("tests", image_id + "-pre.txt")
            result_path = os.path.join("tests", image_id + ".txt")
            if os.path.exists(previous_value_path):
                with open(previous_value_path, "r") as f:
                    previous_value = float(f.read())
            ip = ImageProcessor(image_path, last_valid_config)
            value = ip.process(previous_value)
            with open(result_path, "r") as result_file:
                result = float(result_file.read())
            self.assertEqual(result, value, msg=f"image {image}")
            previous_value = value


class TestDecimalPlaceholderHandling(unittest.TestCase):
    def test_decimal_placeholders_are_sanitized(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        config = {
            "image": {
                "rotate": 0,
                "crop": {"x": 0, "y": 0, "width": 20, "height": 20},
            },
            "digits": [{}],
            "decimal_digits": [{}],
            "decimal_analogs": [],
            "postprocessing": {
                "digits": {"brightness": 0, "contrast": 0},
                "analog": {"brightness": 0, "contrast": 0, "binaryThreshold": 128},
            },
        }

        with mock.patch.object(ImageProcessor, "_parse_digits", side_effect=["12345", "??6"]):
            value = ImageProcessor(encoded.tobytes(), config).process(previous_value=12345.0)

        self.assertEqual(value, 12345.006)


if __name__ == '__main__':
    unittest.main()
