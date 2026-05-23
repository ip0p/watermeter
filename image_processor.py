import math

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import json
import easyocr

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        ocr = easyocr.Reader(["en"], gpu=False)
    return ocr

class ImageProcessor:
    def __init__(self, image_source, config_source):
        """
        image_source: path to an image file (str) OR raw image bytes.
        config_source: path to a JSON config file (str) OR a dict.
        """
        if isinstance(image_source, (bytes, bytearray)):
            arr = np.frombuffer(image_source, dtype=np.uint8)
            self.img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.image_path = None
        else:
            self.image_path = image_source
            self.img = cv2.imread(image_source)

        if isinstance(config_source, dict):
            self.config = config_source
            self.config_path = None
        else:
            self.config_path = config_source
            with open(config_source, "r") as config_file:
                self.config = json.load(config_file)

        if self.img is None:
            raise ValueError("Image not found")

    def process(self, previous_value: None | float = None, debug: str | None = None) -> float:
        return self.process_with_details(previous_value=previous_value, debug=debug)["value"]

    def process_with_details(self, previous_value: None | float = None, debug: str | None = None) -> dict:
        def unpack_parse_result(parse_result):
            if isinstance(parse_result, tuple) and len(parse_result) == 2:
                return parse_result
            return parse_result, []

        # Rotate the image
        (h, w) = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), self.config["image"]["rotate"], 1.0)
        rotated = cv2.warpAffine(self.img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #self.__debug_show_image("rotated", rotated)

        # Crop region of interest
        crop_cfg = self.config["image"]["crop"]
        x, y, cw, ch = crop_cfg["x"], crop_cfg["y"], crop_cfg["width"], crop_cfg["height"]
        cropped = rotated[y:y + ch, x:x + cw]
        #self.__debug_show_image("cropped", cropped)

        # Draw image for debugging
        debug_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(debug_image)

        # Extract digits
        # error handling so debug image still gets created
        err = None
        digits = None
        digit_details = []
        decimal_digits = None
        decimal_digit_details = []
        analog_details = []
        decimal_source = None
        try:
            digits, digit_details = unpack_parse_result(
                self._parse_digits(cropped, draw, debug_image, self.config["digits"], with_details=True)
            )
        except Exception as e:
            if err is None:
                err = e
        try:
            decimal_digits, decimal_digit_details = unpack_parse_result(
                self._parse_digits(cropped, draw, debug_image, self.config["decimal_digits"], with_details=True)
            )
            if decimal_digits is not None:
                decimal_source = "decimal_digits"
        except Exception as e:
            if err is None:
                err = e
        if decimal_digits is None:
            try:
                decimal_digits, analog_details = unpack_parse_result(
                    self._parse_analogs(cropped, draw, debug_image, self.config["decimal_analogs"], with_details=True)
                )
                if decimal_digits is not None:
                    decimal_source = "decimal_analogs"
            except Exception as e:
                if err is None:
                    err = e

        if debug is not None:
            # Save the combined output
            #self.__debug_show_image("digits", combined)
            #self.__debug_show_image("debug", debug_image)
            debug_image.save(debug)
            print("Debug image saved to " + debug)

        if err:
            raise err

        #print(f"Raw results: {digits}.{decimal_digits}")
        digits_sanitized = "".join(c if c.isdigit() else "0" for c in digits)
        final_value = float(digits_sanitized) # if digit parsing fails replace with 0
        if decimal_digits:
            decimal_sanitized = "".join(c if c.isdigit() else "0" for c in decimal_digits)
            decimal_value = float("0." + decimal_sanitized)
            if previous_value:
                # context aware parsing, last digit of the integer value may be wrong due to rotating nature
                # check if we're in range of it being inaccurate and just use last main value
                if decimal_value > 0.2:
                    final_value = math.floor(previous_value)
            final_value += decimal_value

        return {
            "value": final_value,
            "digits": digits,
            "decimal_digits": decimal_digits,
            "digit_details": digit_details,
            "decimal_digit_details": decimal_digit_details,
            "analog_details": analog_details,
            "decimal_source": decimal_source,
        }

    def _parse_digits(self, image, draw: ImageDraw.ImageDraw, debug_image: Image.Image, digit_config: list[dict], with_details: bool = False):
        if len(digit_config) == 0:
            if with_details:
                return None, []
            return None

        FALLBACK_OCR_BORDER_SIZE = 4

        def extract_single_digit(ocr_result):
            best_digit = None
            best_confidence = float("-inf")
            for candidate in ocr_result:
                if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
                    continue
                text = candidate[1]
                if not isinstance(text, str):
                    continue
                digits_only = "".join(c for c in text if c.isdigit())
                if len(digits_only) != 1:
                    continue
                confidence = candidate[2] if len(candidate) >= 3 and isinstance(candidate[2], (int, float)) else float("-inf")
                if confidence > best_confidence:
                    best_digit = digits_only
                    best_confidence = confidence
            return best_digit, best_confidence

        final_text = ""
        details = []
        for d in digit_config:
            dx, dy, dw, dh = d["x"], d["y"], d["width"], d["height"]
            digit_img = image[dy:dy + dh, dx:dx + dw]

            # Convert to PIL for brightness/contrast postprocessing
            digit_pil = Image.fromarray(cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB))

            # Apply brightness / contrast adjustments
            brightness = self.config["postprocessing"]["digits"]["brightness"]
            contrast = self.config["postprocessing"]["digits"]["contrast"]
            if brightness != 0:
                digit_pil = ImageEnhance.Brightness(digit_pil).enhance(1 + brightness / 100)
            if contrast != 0:
                digit_pil = ImageEnhance.Contrast(digit_pil).enhance(1 + contrast / 100)

            # Paste postprocessed digit back so the preview reflects what OCR actually sees
            debug_image.paste(digit_pil, (dx, dy))
            draw.rectangle((dx, dy, dx + dw - 1, dy + dh - 1), outline=(255, 0, 0), width=1)
            #self.__debug_show_image("digit", digit_pil)
            digit_for_ocr = cv2.cvtColor(np.array(digit_pil), cv2.COLOR_RGB2BGR)
            decolor = bool(self.config.get("postprocessing", {}).get("digits", {}).get("decolor", False))

            ocr_candidates = [("normal", digit_for_ocr)]
            if decolor:
                gray = cv2.cvtColor(digit_for_ocr, cv2.COLOR_BGR2GRAY)
                decolored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ocr_candidates.append(("decolor", decolored))

            best_digit = None
            best_confidence = float("-inf")
            best_method = None

            for method, candidate_image in ocr_candidates:
                text = get_ocr().readtext(candidate_image, allowlist="0123456789")
                candidate_digit, confidence = extract_single_digit(text)
                if candidate_digit is not None and confidence > best_confidence:
                    best_digit = candidate_digit
                    best_confidence = confidence
                    best_method = method

            # Fallback: add small border if first pass was ambiguous.
            # This helps narrow glyphs like "1" that can be clipped at crop edges.
            if best_digit is None:
                for method, candidate_image in ocr_candidates:
                    bordered_digit_for_ocr = cv2.copyMakeBorder(
                        candidate_image,
                        FALLBACK_OCR_BORDER_SIZE, FALLBACK_OCR_BORDER_SIZE,
                        FALLBACK_OCR_BORDER_SIZE, FALLBACK_OCR_BORDER_SIZE,
                        cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    text = get_ocr().readtext(bordered_digit_for_ocr, allowlist="0123456789")
                    candidate_digit, confidence = extract_single_digit(text)
                    if candidate_digit is not None and confidence > best_confidence:
                        best_digit = candidate_digit
                        best_confidence = confidence
                        best_method = f"{method}+border"

            if best_digit is None:
                final_text += "?"
                details.append({
                    "x": dx,
                    "y": dy,
                    "width": dw,
                    "height": dh,
                    "digit": "?",
                    "confidence": None,
                    "method": "none",
                })
                continue
            final_text += best_digit
            details.append({
                "x": dx,
                "y": dy,
                "width": dw,
                "height": dh,
                "digit": best_digit,
                "confidence": float(best_confidence),
                "method": best_method,
            })

        if with_details:
            return final_text, details
        return final_text

    def _parse_analogs(self, image, draw: ImageDraw.ImageDraw, debug_image: Image.Image, analogs_config: list[dict], with_details: bool = False) -> str | None:
        if len(analogs_config) == 0:
            if with_details:
                return None, []
            return None
        result = []
        details = []
        for analog in analogs_config:
            value, detail = self._parse_analog(image, draw, debug_image, analog, with_detail=True)
            result.append(value)
            details.append(detail)
        result_str = ""
        # perform context aware parsing
        # this helps with slightly off values due to perspective errors
        for i in range(len(result)):
            if i == len(result) - 1:
                corrected = math.floor(result[i])
                result_str += str(corrected)
                details[i]["corrected_digit"] = corrected
                break

            this_value = result[i]
            next_value = result[i + 1]
            corr_percent = 20
            if this_value % 1 > (1 - corr_percent / 100) and next_value < (corr_percent / 10):
                this_value += (1 - corr_percent / 100)
            elif this_value % 1 < (corr_percent / 100) and next_value > (10 - corr_percent / 10):
                this_value -= (1 - corr_percent / 100)
            this_value = this_value % 10
            corrected = math.floor(this_value)
            result_str += str(corrected)
            details[i]["corrected_digit"] = corrected

        if with_details:
            return result_str, details
        return result_str

    def _parse_analog(self, image, draw: ImageDraw.ImageDraw, debug_image: Image.Image, cfg: dict, with_detail: bool = False):
        dx, dy, dw, dh, color = cfg["x"], cfg["y"], cfg["width"], cfg["height"], cfg["color"]

        analog_image = image[dy:dy + dh, dx:dx + dw]

        # Convert to PIL for brightness/contrast postprocessing
        analog_pil = Image.fromarray(cv2.cvtColor(analog_image, cv2.COLOR_BGR2RGB))

        # Apply brightness / contrast adjustments
        brightness = self.config["postprocessing"]["analog"]["brightness"]
        contrast = self.config["postprocessing"]["analog"]["contrast"]
        if brightness != 0:
            analog_pil = ImageEnhance.Brightness(analog_pil).enhance(1 + brightness / 100)
        if contrast != 0:
            analog_pil = ImageEnhance.Contrast(analog_pil).enhance(1 + contrast / 100)

        # Paste postprocessed analog back so the preview reflects what detection actually uses
        debug_image.paste(analog_pil, (dx, dy))

        draw.rectangle((dx, dy, dx + dw - 1, dy + dh - 1), outline=(255, 0, 0), width=1)
        draw.line((dx, dy, dx + dw - 1, dy + dh - 1), fill=(255, 0, 0), width=1)
        draw.line((dx, dy + dh - 1, dx + dw - 1, dy), fill=(255, 0, 0), width=1)

        analog_image = cv2.cvtColor(np.array(analog_pil), cv2.COLOR_RGB2BGR)
        binary_threshold = self.config["postprocessing"]["analog"]["binaryThreshold"]
        decolor = bool(self.config.get("postprocessing", {}).get("analog", {}).get("decolor", False))

        if decolor:
            gray_image = cv2.cvtColor(analog_image, cv2.COLOR_BGR2GRAY)
            _, target_color_image = cv2.threshold(gray_image, binary_threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            # white/gray to black, isolate colors
            color_index = -1
            if color == "red":
                color_index = 2
            elif color == "green":
                color_index = 1
            elif color == "blue":
                color_index = 0
            min_vals = np.min(analog_image, axis=2, keepdims=True)
            analog_image -= min_vals
            target_color_image = analog_image[:, :, color_index]
            _, target_color_image = cv2.threshold(target_color_image, binary_threshold, 255, cv2.THRESH_BINARY)
        #self.__debug_show_image("analog", target_color_image)

        # now we need to find the white pixel furthest from the center
        height, width = target_color_image.shape
        white_pixels = np.column_stack(np.where(target_color_image == 255))
        if len(white_pixels) == 0:
            threshold = binary_threshold
            raise ValueError(
                f"No '{color}' pointer pixels found after thresholding "
                f"(binaryThreshold={threshold}). "
                "Check that the pointer color matches the 'color' field, try lowering binaryThreshold, "
                "or enable postprocessing.analog.decolor for dark pointers."
            )
        cx, cy = width // 2, height // 2
        distances = np.sqrt((white_pixels[:, 1] - cx) ** 2 + (white_pixels[:, 0] - cy) ** 2)

        furthest_idx = np.argmax(distances)
        py, px = white_pixels[furthest_idx]
        pdx = px - cx
        pdy = py - cy
        angle = (math.degrees(math.atan2(pdy, pdx)) + 90) % 360
        value = angle / 36.0

        draw.line((dx + cx, dy + cy, dx + px, dy + py), fill=(0, 255, 0), width=3)

        def draw_debug_line_for_angle(angle, color):
            angle = angle - 90
            px = dx + cx + math.cos(math.radians(angle)) * cx
            py = dy + cy + math.sin(math.radians(angle)) * cy
            draw.line((dx + cx, dy + cy, px, py), fill=color)

        #for i in range(0, 10):
        #    draw_debug_line_for_angle(i * 36, (0, 0, 100 + i * 10))
        draw_debug_line_for_angle(angle, (255, 255, 0))


        if value < 0 or value >= 10:
            raise ValueError("Invalid angle")
        if with_detail:
            return value, {
                "x": dx,
                "y": dy,
                "width": dw,
                "height": dh,
                "color": color,
                "value": value,
                "digit": math.floor(value),
                "mode": "decolor" if decolor else "color",
            }
        return value

    @staticmethod
    def __debug_show_image(name, image):
        if type(image) == Image.Image:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()