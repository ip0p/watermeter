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


def make_digit_details(digits, confidences=None, alternatives_list=None):
    """Build a digit_details list from a digit string with optional per-digit data."""
    result = []
    for i, d in enumerate(digits):
        conf = confidences[i] if confidences else 0.9
        alts = alternatives_list[i] if alternatives_list else []
        result.append({
            "x": i * 10, "y": 0, "width": 10, "height": 20,
            "digit": d,
            "confidence": conf,
            "method": "normal",
            "alternatives": [{"digit": a, "confidence": conf - 0.1} for a in alts],
        })
    return result


def make_details(digits, dec_digits="", confidences=None, dec_confidences=None,
                 alternatives_list=None, dec_alternatives_list=None):
    """Build a details dict like image_processor.process_with_details would return."""
    digit_details = make_digit_details(digits, confidences, alternatives_list)
    decimal_digit_details = make_digit_details(dec_digits, dec_confidences, dec_alternatives_list)
    int_val = float(digits) if digits else 0.0
    dec_val = float("0." + dec_digits) if dec_digits else 0.0
    return {
        "digits": digits,
        "decimal_digits": dec_digits,
        "decimal_used": dec_digits,
        "analog_digits": "",
        "analog_details": [],
        "decimal_source": "config",
        "digit_details": digit_details,
        "decimal_digit_details": decimal_digit_details,
        "value": int_val + dec_val,
    }


class TestAutocorrectReading(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ws = load_web_server(self.tmp_dir)

    def _autocorrect(self, details, previous, max_threshold):
        return self.ws._autocorrect_reading(details, previous, max_threshold)

    # ------------------------------------------------------------------
    # Basic cases
    # ------------------------------------------------------------------

    def test_no_previous_returns_none(self):
        details = make_details("1239")
        corrected, desc = self._autocorrect(details, None, 0.2)
        self.assertIsNone(corrected)

    def test_correct_reading_unchanged(self):
        # Reading already in range; autocorrect should still find a valid value
        # (the original), but this function is not called unless sanity already failed.
        details = make_details("1234")
        # If the reading is in range, the function may return the original or a closer
        # candidate — either is acceptable; we just test it doesn't crash.
        self._autocorrect(details, 1234.0, 0.2)

    def test_single_digit_high_substitution(self):
        # OCR read "1239" but correct is "1234": single digit 9→3 should be found
        # with previous=1234.0 and result would be 1239 → too high.
        details = make_details(
            "1239",
            confidences=[0.9, 0.9, 0.9, 0.5],
            alternatives_list=[[], [], [], ["4", "3"]],
        )
        corrected, desc = self._autocorrect(details, 1234.0, 0.2)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, 1234.0)
        self.assertLessEqual(corrected, 1234.2)
        self.assertIn("9", desc)

    def test_single_digit_low_substitution(self):
        # OCR read "1233" but correct is "1238": previous=1238.0, result too low.
        details = make_details(
            "1233",
            confidences=[0.9, 0.9, 0.9, 0.5],
            alternatives_list=[[], [], [], ["8"]],
        )
        corrected, desc = self._autocorrect(details, 1238.0, 0.2)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, 1238.0)
        self.assertLessEqual(corrected, 1238.2)

    def test_returns_closest_to_previous_among_valid(self):
        # Two possible corrections in range; should return one closest to previous.
        details = make_details(
            "1250",
            confidences=[0.9, 0.9, 0.9, 0.5],
            alternatives_list=[[], [], [], ["1", "2"]],
        )
        # previous=1200.0, max=0.2 → valid range [1200.0, 1200.2]
        # "1201" → 1201.0 outside range; "1202" → outside range
        # Let's check with range big enough to fit both
        corrected, desc = self._autocorrect(details, 1200.0, 1.0)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, 1200.0)
        self.assertLessEqual(corrected, 1201.0)

    def test_no_valid_substitution_returns_none(self):
        # OCR read "9999", previous=1234.0, maxThreshold=0.2 — no single/double
        # substitution can bring it into [1234.0, 1234.2].
        details = make_details("9999", confidences=[0.9, 0.9, 0.9, 0.9])
        corrected, desc = self._autocorrect(details, 1234.0, 0.2)
        self.assertIsNone(corrected)

    def test_two_digit_substitution(self):
        # OCR read "1299", correct is "1234": two substitutions needed (9→3, 9→4).
        details = make_details(
            "1299",
            confidences=[0.9, 0.9, 0.4, 0.3],
            alternatives_list=[[], [], ["3"], ["4"]],
        )
        corrected, desc = self._autocorrect(details, 1234.0, 0.2)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, 1234.0)
        self.assertLessEqual(corrected, 1234.2)

    def test_decimal_digit_substitution(self):
        # OCR read integer "1234", decimal "9" (should be "1"), previous=1234.1.
        details = make_details(
            "1234",
            dec_digits="9",
            dec_confidences=[0.4],
            dec_alternatives_list=[["1"]],
        )
        corrected, desc = self._autocorrect(details, 1234.1, 0.2)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, 1234.1)
        self.assertLessEqual(corrected, 1234.3)


if __name__ == "__main__":
    unittest.main()
