FROM python:3.11-slim

# x86_64 ONLY — paddle wheels ship AVX builds ("Illegal instruction" on ARM).
# Build with: docker buildx build --platform linux/amd64 .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_CACHE_HOME=/srv/models \
    PADDLEX_HOME=/srv/models

WORKDIR /srv

# System deps: libgl for OpenCV headless quirks, libgomp for PaddlePaddle's
# OpenMP runtime (missing on python:*-slim -> "libgomp.so.1: cannot open
# shared object file"), tesseract only if the fallback is enabled.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# PaddlePaddle CPU wheel first (never the GPU wheel in this image).
RUN pip install --upgrade pip && \
    pip install "paddlepaddle==3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install -r requirements.txt

COPY . .

# Pre-bake OCR model weights so the container runs offline.
# (Slow on first build; comment out for a slim dev image.)
RUN python scripts/download_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
