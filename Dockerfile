FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .

# Install CPU-only PyTorch wheels to avoid CUDA dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Download EasyOCR detection models at build time.
RUN python __main__.py init

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"

VOLUME ["/data"]

EXPOSE 5000

# Default command: run web server
# To use the CLI instead: docker run --rm ... python __main__.py run ...
ENTRYPOINT ["python", "web_server.py"]
