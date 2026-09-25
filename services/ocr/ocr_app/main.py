"""Standalone OCR microservice: wraps PaddleOcrService behind HTTP so it can
run as its own process/container, separate from the detection service.

This separation isn't just architectural taste — PaddleOCR's GPU build
crashes with Windows DLL conflicts when loaded in the same process as
PyTorch (used by the detection service's YOLO model), since both bundle
their own private CUDA/cuDNN redistributables under identical DLL names.
Keeping them in separate processes sidesteps that entirely, and lets this
GPU-bound OCR stage scale independently of the cheap detection stage.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ktp_interfaces import OcrService

from .max_body_size_middleware import MaxBodySizeMiddleware

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Optional shared-secret check between this service and its only intended
# caller (the api service's RemoteOcrService, which sends the same value as
# an X-Internal-Key header when KTP_OCR_INTERNAL_KEY is set on that side
# too). Off by default — nothing in this repo's deployment config exposes
# this service's port beyond localhost today — but this service otherwise
# has no auth of its own at all, so this is a free defense-in-depth layer
# for whenever that changes (e.g. a shared docker-compose network).
# `or None`: an empty value (e.g. a blank .env entry, which docker-compose
# passes through as "") means "not set", not "require an empty key".
_INTERNAL_KEY = os.getenv("KTP_OCR_INTERNAL_KEY") or None


async def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    if _INTERNAL_KEY is None:
        return  # disabled (dev only / no untrusted network path to this service yet)
    if x_internal_key is None or not hmac.compare_digest(x_internal_key, _INTERNAL_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid internal key")


@lru_cache(maxsize=1)
def get_ocr_service() -> OcrService:
    """Built once, cached. Kept behind a FastAPI dependency (rather than a
    bare module-level singleton) so tests can override it with a fake and
    never load a real model.

    Defaults to the native PaddlePaddle engine (unchanged behavior). Set
    KTP_OCR_ENGINE=onnxruntime plus KTP_OCR_ONNX_DET_DIR/KTP_OCR_ONNX_REC_DIR
    (pointing at model directories already converted via
    `paddlex --paddle2onnx`) to use the ONNX Runtime path instead — see
    PaddleOcrService and docs/CHANGELOG.md for the measured tradeoffs.
    """
    from ktp_ocr import PaddleOcrService

    engine = os.getenv("KTP_OCR_ENGINE", "paddle")
    if engine == "onnxruntime":
        return PaddleOcrService(
            engine="onnxruntime",
            text_detection_model_dir=os.environ["KTP_OCR_ONNX_DET_DIR"],
            text_recognition_model_dir=os.environ["KTP_OCR_ONNX_REC_DIR"],
        )
    return PaddleOcrService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model up at startup rather than on the first request — a
    # cold first request paid the full ~17s model-load cost on top of
    # inference, which whichever caller hit the service first would eat.
    # Goes through dependency_overrides so tests (which override
    # get_ocr_service with a fake before the app starts) never trigger a
    # real PaddleOCR load here.
    factory = app.dependency_overrides.get(get_ocr_service, get_ocr_service)
    factory()
    yield


app = FastAPI(
    title="KTP OCR Service",
    version="0.1.0",
    description="Extracts KTP fields from an already-detected, rectified card image.",
    lifespan=lifespan,
)

# Rejects an oversized request body at the ASGI layer, before FastAPI's own
# multipart parsing (which applies no size cap to file parts) ever reads
# it — see max_body_size_middleware.py for why enforcing this inside the
# endpoint, after the body is already fully buffered, is too late. Matters
# more here than on the api service since this service has no auth at all.
app.add_middleware(MaxBodySizeMiddleware, max_body_size=MAX_UPLOAD_BYTES)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/ocr/extract", dependencies=[Depends(require_internal_key)])
async def extract(file: UploadFile, ocr_service: OcrService = Depends(get_ocr_service)):
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes"
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Could not decode uploaded image") from exc
    except Image.DecompressionBombError as exc:
        # Pillow's own default limit (Image.MAX_IMAGE_PIXELS) already
        # prevents the actual memory-exhaustion risk here — this is just
        # making sure a maliciously huge image gets the same clean 422 any
        # other bad upload gets, not an unhandled 500. Mirrors the same fix
        # already applied to services/api/app/pipeline.py.
        raise HTTPException(status_code=422, detail="Could not decode uploaded image") from exc

    # extract_fields() is synchronous and does real CPU-bound OCR inference
    # (multiple seconds) — run it off the event loop so one request doesn't
    # block every other request this process is serving, including health
    # checks.
    fields = await asyncio.to_thread(ocr_service.extract_fields, image)
    return fields.model_dump(mode="json")
