import unittest
from glob import glob
import os
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw

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
            try:
                value = ip.process(previous_value)
            except RuntimeError as exc:
                if "Download from" in str(exc) and "failed. Retry limit reached" in str(exc):
                    self.skipTest("PaddleOCR model download unavailable in this environment")
                raise
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

    def test_decimal_placeholders_fall_back_to_analog(self):
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
            "decimal_analogs": [{}],
            "postprocessing": {
                "digits": {"brightness": 0, "contrast": 0},
                "analog": {"brightness": 0, "contrast": 0, "binaryThreshold": 128},
            },
        }

        with mock.patch.object(ImageProcessor, "_parse_digits", side_effect=["12345", "??"]):
            with mock.patch.object(ImageProcessor, "_parse_analogs", return_value="6789"):
                details = ImageProcessor(encoded.tobytes(), config).process_with_details(previous_value=12345.0)

        self.assertEqual(details["value"], 12345.6789)
        self.assertEqual(details["decimal_source"], "decimal_analogs")
        self.assertEqual(details["analog_digits"], "6789")


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
        mock_reader.ocr.return_value = [[
            ("?", 0.99),
            ("4", 0.80),
            ("8", 0.20),
        ]]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 4.0)

    def test_accepts_digit_embedded_in_single_character_noise(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        mock_reader.ocr.return_value = [[
            ("7.", 0.90),
        ]]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 7.0)

    def test_invalid_candidates_fall_back_to_zero(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        mock_reader.ocr.return_value = [[
            ("??", 0.90),
            ("", 0.80),
            ("AB", 0.70),
        ]]

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            value = ip.process()

        self.assertEqual(value, 0.0)

    def test_scaled_fallback_can_recover_missed_digit(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        # Base test digit crop is 20x20; fallback adds 4px border each side (28x28),
        # so 56x56 indicates the 2x scaled fallback variant.
        min_scaled_dimension = 56
        low_confidence = 0.40
        recovered_confidence = 0.94

        def ocr_side_effect(image, det=False, rec=True, cls=False):
            if image.shape[0] >= min_scaled_dimension and image.shape[1] >= min_scaled_dimension:
                return [[("9", recovered_confidence)]]
            return [[("??", low_confidence)]]

        mock_reader.ocr.side_effect = ocr_side_effect

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            details = ip.process_with_details()

        self.assertEqual(details["value"], 9.0)
        self.assertEqual(details["digit_details"][0]["digit"], "9")
        self.assertIn("scale", details["digit_details"][0]["method"])

    def test_three_x_scaled_fallback_can_recover_missed_digit(self):
        ip = self._build_processor()
        mock_reader = mock.Mock()
        # Base crop 20x20 -> bordered 28x28 -> 3x scaled is 84x84.
        min_three_x_scaled_dimension = 84
        low_confidence = 0.40
        recovered_confidence = 0.95

        def ocr_side_effect(image, det=False, rec=True, cls=False):
            if image.shape[0] >= min_three_x_scaled_dimension and image.shape[1] >= min_three_x_scaled_dimension:
                return [[("8", recovered_confidence)]]
            return [[("??", low_confidence)]]

        mock_reader.ocr.side_effect = ocr_side_effect

        with mock.patch("image_processor.get_ocr", return_value=mock_reader):
            details = ip.process_with_details()

        self.assertEqual(details["value"], 8.0)
        self.assertEqual(details["digit_details"][0]["digit"], "8")
        self.assertIn("scale3x", details["digit_details"][0]["method"])


class TestAnalogPointerDetection(unittest.TestCase):
    def _build_processor(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        config = {
            "image": {
                "rotate": 0,
                "crop": {"x": 0, "y": 0, "width": 120, "height": 120},
            },
            "digits": [],
            "decimal_digits": [],
            "decimal_analogs": [],
            "postprocessing": {
                "digits": {"brightness": 0, "contrast": 0},
                "analog": {"brightness": 0, "contrast": 0, "binaryThreshold": 120},
            },
        }
        return ImageProcessor(encoded.tobytes(), config)

    def _parse_single_analog(self, processor, analog_image):
        debug_image = Image.fromarray(cv2.cvtColor(analog_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(debug_image)
        value, details = processor._parse_analog(
            analog_image,
            draw,
            debug_image,
            {"x": 0, "y": 0, "width": analog_image.shape[1], "height": analog_image.shape[0], "color": "red"},
            with_detail=True,
        )
        return value, details

    def test_prefers_center_connected_pointer_component(self):
        processor = self._build_processor()
        analog_image = np.zeros((120, 120, 3), dtype=np.uint8)
        center = (60, 60)

        # Actual pointer (center-connected).
        cv2.line(analog_image, center, (35, 95), (0, 0, 255), 4)
        pointer_value, _ = self._parse_single_analog(processor, analog_image.copy())

        # Add unrelated far-edge red blob that should be ignored.
        cv2.circle(analog_image, (5, 5), 8, (0, 0, 255), -1)
        noisy_value, _ = self._parse_single_analog(processor, analog_image)

        # Two decimals are sufficient here: the assertion guards against major direction changes,
        # not tiny pixel-level tip jitter from rasterization.
        self.assertAlmostEqual(pointer_value, noisy_value, places=2)

    def test_falls_back_when_no_center_connected_component_exists(self):
        processor = self._build_processor()
        analog_image = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.circle(analog_image, (10, 10), 8, (0, 0, 255), -1)

        value, details = self._parse_single_analog(processor, analog_image)

        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 10)
        self.assertEqual(details["mode"], "color")


if __name__ == '__main__':
    unittest.main()
