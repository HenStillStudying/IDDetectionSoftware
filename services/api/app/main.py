from __future__ import annotations

import logging
import os

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ktp_interfaces import DetectionService, OcrService
from ktp_schema import ExtractionStatus

from .config import settings
from .jobs import JobStatus, JobStore
from .pipeline import KtpExtractionPipeline
from .remote_ocr_service import RemoteOcrService
from .stub_models import StubDetectionService, StubOcrService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="KTP Identification API",
    version="0.1.0",
    description="Detects an Indonesian KTP card in a photo and extracts its fields via OCR.",
)


def _build_detection_service() -> DetectionService:
    """Uses the trained YOLO detector when KTP_DETECTION_WEIGHTS points at a
    real weights file; falls back to the stub (0 confidence, whole image as
    the card) otherwise, so the service still boots in dev/CI without a
    trained model.
    """
    weights_path = settings.detection_weights_path
    if weights_path and os.path.exists(weights_path):
        from ktp_detection import YoloDetectionService

        logger.info("Loading YOLO detection model from %s", weights_path)
        return YoloDetectionService(weights_path)

    logger.warning("KTP_DETECTION_WEIGHTS not set or missing — using stub detection service")
    return StubDetectionService()


def _build_ocr_service() -> OcrService:
    """Calls the separately-deployed OCR microservice (services/ocr) over
    HTTP when KTP_OCR_SERVICE_URL is set; falls back to the stub otherwise,
    so the service still boots without that dependency running.

    OCR runs as its own service (not imported in-process here) because
    PaddleOCR's GPU build crashes with Windows DLL conflicts when loaded
    alongside PyTorch (used by the detection service below) — see
    services/ocr's README for the full story.
    """
    if settings.ocr_service_url:
        logger.info("Using remote OCR service at %s", settings.ocr_service_url)
        return RemoteOcrService(settings.ocr_service_url)

    logger.warning("KTP_OCR_SERVICE_URL not set — using stub OCR service")
    return StubOcrService()


pipeline = KtpExtractionPipeline(
    detection_service=_build_detection_service(),
    ocr_service=_build_ocr_service(),
)
job_store = JobStore()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return  # auth disabled (dev only)
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_bytes} bytes",
        )
    return contents


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/ktp/extract", dependencies=[Depends(require_api_key)])
async def extract(file: UploadFile):
    contents = await _read_and_validate_upload(file)
    result = pipeline.run(contents)

    status_code = 200
    if result.status == ExtractionStatus.INVALID_IMAGE:
        status_code = 422
    elif result.status == ExtractionStatus.NO_CARD_DETECTED:
        status_code = 404

    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


def _run_job(job_id: str, contents: bytes) -> None:
    job_store.mark_processing(job_id)
    try:
        result = pipeline.run(contents)
        job_store.mark_done(job_id, result)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure on the job
        job_store.mark_failed(job_id, str(exc))


@app.post("/v1/ktp/jobs", dependencies=[Depends(require_api_key)])
async def create_job(file: UploadFile, background_tasks: BackgroundTasks):
    contents = await _read_and_validate_upload(file)
    job = job_store.create()
    background_tasks.add_task(_run_job, job.id, contents)
    return {"job_id": job.id, "status": job.status}


@app.get("/v1/ktp/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job.id, "status": job.status}
    if job.status == JobStatus.DONE:
        response["result"] = job.result.model_dump(mode="json")
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error
    return response
