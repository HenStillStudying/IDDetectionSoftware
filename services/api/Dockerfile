FROM python:3.12-slim

WORKDIR /app

# Build context must be the repo root: docker build -f services/api/Dockerfile .
COPY libs ./libs
RUN pip install --no-cache-dir ./libs

# Detection runs in-process inside this service (cheap — a YOLOv8n forward
# pass is milliseconds — unlike OCR, which is its own service). Without
# this, ktp_detection is never installed, pipeline_factory.py's
# `from ktp_detection import YoloDetectionService` fails at import time,
# and the container silently falls back to stub detection even with real
# weights mounted in.
#
# torch/torchvision installed first from PyTorch's CPU-only index, pinned to
# the same versions used locally: ultralytics would otherwise pull the
# default PyPI build, which bundles several GB of CUDA libraries this
# CPU-only container never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1
COPY services/detection/ktp_detection ./ktp_detection
COPY services/detection/pyproject.toml .
RUN pip install --no-cache-dir .

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ultralytics pulls in the full (GUI) opencv-python alongside the headless
# build services/detection pins, and the full one wins the shared cv2
# import — it needs these system libs, which python:3.12-slim doesn't ship
# (without them, `import cv2` fails with libxcb.so.1 / libGL.so.1 missing).
# Placed after the pip layers so changing it doesn't invalidate them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/api/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
