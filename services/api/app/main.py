from __future__ import annotations

from arq.jobs import Job, JobStatus
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ktp_schema import ExtractionStatus

from .config import settings
from .pipeline_factory import build_pipeline
from .redis_pool import get_redis_pool

app = FastAPI(
    title="KTP Identification API",
    version="0.1.0",
    description="Detects an Indonesian KTP card in a photo and extracts its fields via OCR.",
)

# Used by the sync endpoint only. The async endpoints enqueue work onto
# Redis instead — see app/worker.py for the process that actually runs the
# pipeline for those.
pipeline = build_pipeline()


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


@app.post("/v1/ktp/jobs", dependencies=[Depends(require_api_key)])
async def create_job(file: UploadFile, redis=Depends(get_redis_pool)):
    contents = await _read_and_validate_upload(file)
    job = await redis.enqueue_job("process_ktp_extraction", contents)
    return {"job_id": job.job_id, "status": JobStatus.queued.value}


@app.get("/v1/ktp/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str, redis=Depends(get_redis_pool)):
    job = Job(job_id, redis)
    status = await job.status()
    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": status.value}
    if status == JobStatus.complete:
        try:
            response["result"] = await job.result(timeout=5)
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure on the job
            response["status"] = "failed"
            response["error"] = str(exc)
    return response
