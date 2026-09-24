FROM python:3.12-slim

WORKDIR /app

# Build context must be the repo root: docker build -f services/api/Dockerfile .
#
# Layers go from least to most often changed: system libs, then every
# third-party dependency (installed from the dependency manifests alone),
# then this repo's own code last. Editing libs/, services/detection/ or
# app code then only rebuilds the final few small layers, instead of
# re-downloading PyTorch and friends.

# ultralytics pulls in the full (GUI) opencv-python alongside the headless
# build services/detection pins, and the full one wins the shared cv2
# import — it needs these system libs, which python:3.12-slim doesn't ship
# (without them, `import cv2` fails with libxcb.so.1 / libGL.so.1 missing).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision first, from PyTorch's CPU-only index, pinned to the
# same versions used locally: ultralytics would otherwise pull the default
# PyPI build, which bundles several GB of CUDA libraries this CPU-only
# container never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1

# Every other third-party dependency, read from the manifests only — no
# source code copied yet. Detection runs in-process inside this service
# (cheap — a YOLOv8n forward pass is milliseconds — unlike OCR, which is
# its own service), so its dependencies belong here too.
COPY infra/pyproject_deps.py /tmp/deps/
COPY libs/pyproject.toml /tmp/deps/libs.toml
COPY services/detection/pyproject.toml /tmp/deps/detection.toml
COPY services/api/requirements.txt /tmp/deps/
RUN python /tmp/deps/pyproject_deps.py /tmp/deps/libs.toml /tmp/deps/detection.toml > /tmp/deps/pyproject.txt \
    && pip install --no-cache-dir -r /tmp/deps/pyproject.txt -r /tmp/deps/requirements.txt \
    && rm -rf /tmp/deps

# This repo's own packages — --no-deps, since everything they depend on is
# already installed above. Without ktp_detection, pipeline_factory.py's
# `from ktp_detection import YoloDetectionService` fails and the container
# silently falls back to stub detection even with real weights mounted in.
COPY libs ./libs
RUN pip install --no-cache-dir --no-deps ./libs
COPY services/detection/ktp_detection ./ktp_detection
COPY services/detection/pyproject.toml .
RUN pip install --no-cache-dir --no-deps .

COPY services/api/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
