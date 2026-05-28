import math

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import json
from paddleocr import PaddleOCR

ocr_readers = {}
DIAL_POINTER_CENTER_TOLERANCE_RATIO = 0.08
DIAL_POINTER_CENTER_TOLERANCE_MIN_PX = 3
DIAL_POINTER_DILATION_KERNEL_SIZE = 3
DIAL_POINTER_TIP_DISTANCE_RATIO = 0.92
DIAL_POINTER_DILATION_KERNEL = np.ones(
    (DIAL_POINTER_DILATION_KERNEL_SIZE, DIAL_POINTER_DILATION_KERNEL_SIZE),
    dtype=np.uint8,
)

def _normalize_ocr_language(language: str | None = None) -> str:
    normalized = (language or "en").strip().lower()
    if normalized in ("standard", "english", "english_g2"):
        return "en"
    return normalized


def get_ocr(language: str | None = None):
    language_code = _normalize_ocr_language(language)
    if language_code not in ocr_readers:
        ocr_readers[language_code] = PaddleOCR(
            use_angle_cls=False,
            lang=language_code,
            use_gpu=False,
            show_log=False,
        )
    return ocr_readers[language_code]


def _normalize_paddleocr_result(result):
    flattened = []
    if not isinstance(result, list):
        return flattened
    for block in result:
        if isinstance(block, tuple):
            block = [block]
        if not isinstance(block, list):
            continue
        for candidate in block:
            if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
                continue
            if isinstance(candidate[0], str):
                text = candidate[0]
                confidence = candidate[1]
            elif (
                isinstance(candidate[1], (list, tuple))
                and len(candidate[1]) >= 2
                and isinstance(candidate[1][0], str)
            ):
                text = candidate[1][0]
                confidence = candidate[1][1]
            else:
                continue
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            flattened.append([None, text, confidence])
    return flattened

class ImageProcessor:
    def __init__(self, image_source, config_source, ocr_model: str | None = None):
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
        self.ocr_model = _normalize_ocr_language(ocr_model)

    def process(self, previous_value: None | float = None, debug: str | None = None) -> float:
        return self.process_with_details(previous_value=previous_value, debug=debug)["value"]

    def process_with_details(self, previous_value: None | float = None, debug: str | None = None) -> dict:
        def unpack_parse_result(parse_result):
            if (
                isinstance(parse_result, tuple)
                and len(parse_result) == 2
                and isinstance(parse_result[1], list)
            ):
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
        analog_digits = None
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
        try:
            analog_digits, analog_details = unpack_parse_result(
                self._parse_analogs(cropped, draw, debug_image, self.config["decimal_analogs"], with_details=True)
            )
        except Exception as e:
            if err is None:
                err = e

        selected_decimal_digits = decimal_digits
        has_decimal_noise = (
            selected_decimal_digits is not None
            and any(not c.isdigit() for c in selected_decimal_digits)
        )
        if (selected_decimal_digits is None or has_decimal_noise) and analog_digits is not None:
            selected_decimal_digits = analog_digits
            decimal_source = "decimal_analogs"

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
        if selected_decimal_digits:
            decimal_sanitized = "".join(c if c.isdigit() else "0" for c in selected_decimal_digits)
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
            "analog_digits": analog_digits,
            "decimal_used": selected_decimal_digits,
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
        # 2x and 3x keep small wheel glyphs readable for OCR while avoiding excessive blur/noise.
        FALLBACK_OCR_SCALE_FACTORS = (2, 3)
        # 0.9 is treated as "confident enough" to stop retrying transformed variants for this digit.
        FALLBACK_EARLY_EXIT_CONFIDENCE = 0.9

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

        def run_ocr(candidate_image):
            result = get_ocr(self.ocr_model).ocr(candidate_image, det=False, rec=True, cls=False)
            return _normalize_paddleocr_result(result)

        def build_fallback_images(candidate_image):
            bordered = cv2.copyMakeBorder(
                candidate_image,
                FALLBACK_OCR_BORDER_SIZE, FALLBACK_OCR_BORDER_SIZE,
                FALLBACK_OCR_BORDER_SIZE, FALLBACK_OCR_BORDER_SIZE,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            fallback_images = [("border", bordered)]
            for scale in FALLBACK_OCR_SCALE_FACTORS:
                resized = cv2.resize(
                    bordered,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                fallback_images.append((f"border+scale{scale}x", resized))

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                fallback_images.append((f"border+scale{scale}x+otsu", thresh_bgr))
            return fallback_images

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
                text = run_ocr(candidate_image)
                candidate_digit, confidence = extract_single_digit(text)
                if candidate_digit is not None and confidence > best_confidence:
                    best_digit = candidate_digit
                    best_confidence = confidence
                    best_method = method

            # Fallback: add small border if first pass was ambiguous.
            # This helps narrow glyphs like "1" that can be clipped at crop edges.
            if best_digit is None:
                for method, candidate_image in ocr_candidates:
                    for fallback_suffix, fallback_image in build_fallback_images(candidate_image):
                        text = run_ocr(fallback_image)
                        candidate_digit, confidence = extract_single_digit(text)
                        if candidate_digit is not None and confidence > best_confidence:
                            best_digit = candidate_digit
                            best_confidence = confidence
                            best_method = f"{method}+{fallback_suffix}"
                            if best_confidence >= FALLBACK_EARLY_EXIT_CONFIDENCE:
                                break
                    if best_confidence >= FALLBACK_EARLY_EXIT_CONFIDENCE:
                        break

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
            else:
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
        # but only from the component that is connected close to the center
        # so markings/noise at the border are ignored.
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
        # Pointer strokes should pass near the dial center, while labels/noise near borders should not.
        # 8% (~13px on typical 160x160 dials) with a 3px floor keeps slight center offsets,
        # but still rejects markings that are clearly detached from the spindle area.
        center_tolerance = max(
            DIAL_POINTER_CENTER_TOLERANCE_MIN_PX,
            min(width, height) * DIAL_POINTER_CENTER_TOLERANCE_RATIO,
        )
        component_image = cv2.dilate(
            target_color_image,
            # 3x3 dilation bridges 1px gaps from threshold noise while avoiding broad merges
            # that larger kernels could introduce between pointer and nearby markings.
            DIAL_POINTER_DILATION_KERNEL,
            iterations=1,
        )
        num_labels, labels = cv2.connectedComponents(component_image)
        white_pixel_labels = labels[white_pixels[:, 0], white_pixels[:, 1]]

        pointer_pixels = white_pixels
        best_component_label = None
        best_component_reach = -1.0
        for label in range(1, num_labels):
            component_pixels = white_pixels[white_pixel_labels == label]
            if len(component_pixels) == 0:
                continue
            component_distances = np.sqrt(
                (component_pixels[:, 1] - cx) ** 2 + (component_pixels[:, 0] - cy) ** 2
            )
            min_distance = float(np.min(component_distances))
            if min_distance > center_tolerance:
                continue
            reach = float(np.max(component_distances))
            if reach > best_component_reach:
                best_component_reach = reach
                best_component_label = label

        if best_component_label is not None:
            pointer_pixels = white_pixels[white_pixel_labels == best_component_label]

        distances = np.sqrt((pointer_pixels[:, 1] - cx) ** 2 + (pointer_pixels[:, 0] - cy) ** 2)
        # Use the outermost ~8% of pointer pixels and average them to reduce jitter from single-pixel noise.
        tip_distance_threshold = float(np.max(distances)) * DIAL_POINTER_TIP_DISTANCE_RATIO
        tip_pixels = pointer_pixels[distances >= tip_distance_threshold]
        if len(tip_pixels) == 0:
            tip_pixels = pointer_pixels[[np.argmax(distances)]]
        py, px = np.round(np.mean(tip_pixels, axis=0)).astype(int)
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