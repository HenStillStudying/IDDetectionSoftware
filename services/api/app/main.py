from __future__ import annotations

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ktp_schema import ExtractionStatus

from .config import settings
from .jobs import JobStatus, JobStore
from .pipeline import KtpExtractionPipeline
from .stub_models import StubDetectionService, StubOcrService

app = FastAPI(
    title="KTP Identification API",
    version="0.1.0",
    description="Detects an Indonesian KTP card in a photo and extracts its fields via OCR.",
)

# Wired to stubs for now — swap for real model-serving clients in
# services/detection and services/ocr once trained.
pipeline = KtpExtractionPipeline(
    detection_service=StubDetectionService(),
    ocr_service=StubOcrService(),
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
