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

import io
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ktp_interfaces import OcrService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@lru_cache(maxsize=1)
def get_ocr_service() -> OcrService:
    """Built once, cached. Kept behind a FastAPI dependency (rather than a
    bare module-level singleton) so tests can override it with a fake and
    never load a real model.
    """
    from ktp_ocr import PaddleOcrService

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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/ocr/extract")
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

    fields = ocr_service.extract_fields(image)
    return fields.model_dump(mode="json")
