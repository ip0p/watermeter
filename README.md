# 💧 WaterMeter (EasyOCR Edition)

A **water meter reader** inspired by https://github.com/nohn/watermeter that uses **EasyOCR** (instead of Tesseract) to recognize digits on analog water meters. EasyOCR offers improved accuracy and robustness for digit recognition.

This project processes an image of your water meter, crops and corrects it based on a configuration file, extracts the relevant digit and analog fields, and produces a reliable reading with built-in sanity checks.

---

## 🚀 Features

* 🧠 Uses **EasyOCR** for digit recognition (better than Tesseract for uneven lighting or low contrast).
* 🔄 **Context-Aware Parsing:** Combines OCR results with analog dial analysis to auto-correct small visual or mechanical errors between integer and decimal parts.
* 🔍 Supports **cropping and rotation** of the image to isolate the meter area.
* ⚙️ Configurable for different meter layouts via a JSON config file.
* 🧮 Includes **sanity checking**: ensures readings are consistent and realistic.
* 🐳 Easily run anywhere via **Docker**, no Python setup required.
* 🌐 Built-in **web UI** and **REST API** for easy configuration and on-demand readings.

---

## 🐳 Run via Docker (Web UI + REST API)

```bash
docker run -d \
  --name watermeter \
  -p 5000:5000 \
  -v $PWD/data:/data \
  ghcr.io/ip0p/watermeter:latest
```

Then open **http://localhost:5000** in your browser.

The `/data` volume holds:

| File | Description |
|---|---|
| `config.json` | Meter image processing config (crop, rotation, digit positions) |
| `settings.json` | App settings (image URL, max threshold) |
| `value.txt` | Last known meter reading (used for sanity checks) |

The published image is built for both `linux/amd64` and `linux/arm64` (Raspberry Pi 4).

---

## 🧭 Deploy in Portainer (Stack)

1. Go to **Stacks** → **Add stack**.
2. Set stack name to `watermeter`.
3. Paste the compose content from `docker-compose.portainer.yml` in the repository root:

```yaml
version: "3.8"

services:
  watermeter:
    image: ghcr.io/ip0p/watermeter:latest
    container_name: watermeter
    ports:
      - "5000:5000"
    volumes:
      - ${WATERMETER_DATA_DIR:-/opt/watermeter-data}:/data
    restart: unless-stopped
```

4. Deploy the stack.
5. Initialize the mounted data directory (`/opt/watermeter-data`):

```bash
sudo mkdir -p /opt/watermeter-data
curl -fsSL https://raw.githubusercontent.com/ip0p/watermeter/main/config-example.json \
  | sudo tee /opt/watermeter-data/config.json >/dev/null
cat <<'JSON' | sudo tee /opt/watermeter-data/settings.json >/dev/null
{
  "imageUrl": "http://camera/snapshot.jpg",
  "maxThreshold": 0.2
}
JSON
echo "12345.0000" | sudo tee /opt/watermeter-data/value.txt >/dev/null
```

Replace `12345.0000` with your current real meter value.

6. Open `http://<your-server>:5000`, then check/save **Settings** and **Processor Config**.
7. Test with **Read Now** or via API:

```bash
curl -X POST http://<your-server>:5000/api/read \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## 🌐 Web Interface

Open `http://localhost:5000` to access the web UI:

* **Dashboard** — Shows the current reading and a "Read Now" button. Displays the annotated debug image after each read.
* **Settings** — Configure the image snapshot URL and max threshold.
* **Processor Config** — Edit `config.json` directly in the browser.

---

## 🔌 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/value` | Return the last stored meter reading |
| `POST` | `/api/read` | Fetch image from configured URL, run OCR, return reading + debug image |
| `GET` | `/api/settings` | Return current app settings |
| `PUT` | `/api/settings` | Update app settings |
| `GET` | `/api/config` | Return current processor config |
| `PUT` | `/api/config` | Update processor config |

### `POST /api/read`

You may optionally override the image URL per request:

```json
{ "imageUrl": "http://camera/snapshot.jpg" }
```

Response:

```json
{
  "value": 12345.1234,
  "previous": 12345.0000,
  "debugImage": "data:image/jpeg;base64,..."
}
```

On error (sanity check failure, fetch error, etc.) an HTTP 4xx/5xx is returned:

```json
{ "error": "Result 12345.0 is less than previous 12345.1" }
```

---

## 🔧 General setup (cron / headless)

This tool can also run in a minutely cronjob in CLI mode (backward-compatible):

```bash
# Download a frame from an RTSP stream
ffmpeg -rtsp_transport tcp -i "rtsp://RTSP_STREAM_URL" -frames:v 1 -update 1 -q:v 2 -y /output/path/for/image.png

# Process it
docker run --rm -v $PWD:/app/data ghcr.io/ip0p/watermeter \
  python __main__.py run \
  --image data/image.png \
  --config data/config.json \
  --value data/result.txt
```

---

## 🧩 Example Configuration

Create a file named `config.json` based off the [configuration example in this repository](./config-example.json). Also create a `value.txt` file with your current meter reading for context aware parsing.

---

## ⚙️ Configuration Explained

| Section             | Key            | Description                                                                                                                                     |
| ------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **sanity**          | `maxThreshold` | Maximum allowed increase in readings between runs (e.g., `0.2` means next reading can't jump by more than 0.2 m³). Prevents OCR mistakes.       |
| **image**           | `rotate`       | Rotates the image (in degrees) to align the meter horizontally.                                                                                 |
|                     | `crop`         | Defines the rectangular region containing the entire meter display. Coordinates are relative to the original image.                             |
| **digits**          | list           | Each entry defines the x/y coordinates and width/height of an individual digit field in the main counter.                                       |
| **decimal_digits**  | list           | Optional fields for fractional digits (if present on your meter). Empty in this example.                                                        |
| **decimal_analogs** | list           | Circular or analog dials representing decimal fractions. Each field defines the area to analyze and its color channel (`red`, `green`, `blue`). |
| **postprocessing**  | `digits`       | Adjust brightness and contrast to improve OCR results for the main digits. Values are percentages (`-30` = darker, `40` = more contrast).       |
|                     | `analog`       | Same as above, but for analog (dial) sections. `binaryThreshold` defines the grayscale threshold used for detecting pointer position.           |

---

## 🧠 OCR Engine: EasyOCR

This version uses [**EasyOCR**](https://github.com/JaidedAI/EasyOCR), which is built on PyTorch and provides:

* Better character recognition under poor lighting
* Support for multiple fonts and rotated digits
* Fast and accurate performance on CPU (no CUDA required)

## 🧠 Context-Aware Parsing

Reading analog water meters isn't always straightforward - small perspective distortions, glare, or dial overlaps can cause subtle OCR errors.
To improve accuracy, this project uses **context-aware logic** that considers relationships between multiple readings instead of treating each digit or dial in isolation.

### 🔹 Analog Dial Correction

For meters with several rotating dials (the *decimal_analogs* section in your config), the system compares the detected pointer angles across all dials.
Because analog dials are mechanically linked, a small offset on one dial (e.g., the pointer slightly before or after a number) can be corrected by analyzing the adjacent dials' positions.
This significantly reduces false readings caused by:

* Camera perspective skew
* Shadows or glare on one dial
* Slight pointer misalignment

### 🔹 Integer–Decimal Consistency

Analog water meters often have **rolling digit wheels**, where the last (rightmost) digit is partially rotated when transitioning to the next value.
To handle this, the integer OCR results are **cross-checked against the decimal dials**:

* If the decimal dial indicates that a rollover is happening (e.g., between 0.2 and 1), the last integer digit is ignored and the previous value is used.
* This prevents misreads such as `"12345"` when the true reading is `"12344.9"`.
* This also prevents misreads from partially rotated digits such as `12346.4` when OCR detects the `6` as `0` because it is partially cut off

This context-aware correction makes the final reading far more reliable than pure OCR alone.

---

## 🧪 Development Setup (Optional)

If you prefer to run it locally (without Docker):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python web_server.py          # starts web UI on http://localhost:5000
# or
python main.py run --image tests/0001.png --config tests/0001.json --value tests/0001.txt
```
