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


class TestDigitExtractionRobustness(unittest.TestCase):
    def _build_processor(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        config = {
            "image": {
                "rotate": 0,
                "crop": {"x": 0, "y": 0, "width": 20, "height": 20},
            },
            "digits": [{"x": 0, "y": 0, "width": 20, "height": 20}],
            "decimal_digits": [],
            "decimal_analogs": [],
            "postprocessing": {
                "digits": {"brightness": 0, "contrast": 0},
                "analog": {"brightness": 0, "contrast": 0, "binaryThreshold": 128},
            },
        }
        return ImageProcessor(encoded.tobytes(), config)

    def test_prefers_highest_confidence_single_digit_candidate(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [
            [None, "?", 0.99],
            [None, "4", 0.80],
            [None, "8", 0.20],
        ]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 4.0)

    def test_accepts_digit_embedded_in_single_character_noise(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [
            [None, "7.", 0.90],
        ]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 7.0)

    def test_invalid_candidates_fall_back_to_zero(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [
            [None, "??", 0.90],
            [None, "", 0.80],
            [None, "AB", 0.70],
        ]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 0.0)

    def test_scaled_fallback_can_recover_missed_digit(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        # Base test digit crop is 20x20; fallback adds 4px border each side (28x28),
        # so 56x56 indicates at least the 2x scaled fallback variant.
        min_scaled_dimension = 56
        low_confidence = 0.40
        recovered_confidence = 0.94

        def readtext_side_effect(image, allowlist=None):
            if image.shape[0] >= min_scaled_dimension and image.shape[1] >= min_scaled_dimension:
                return [[None, "9", recovered_confidence]]
            return [[None, "??", low_confidence]]

        mock_reader.readtext.side_effect = readtext_side_effect

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            details = ip.process_with_details()

        self.assertEqual(details["value"], 9.0)
        self.assertEqual(details["digit_details"][0]["digit"], "9")
        self.assertIn("scale", details["digit_details"][0]["method"])


if __name__ == '__main__':
    unittest.main()
