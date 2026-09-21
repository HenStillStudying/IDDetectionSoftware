from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path

from arq.jobs import Job, JobStatus
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from ktp_schema import ExtractionStatus

from .config import settings
from .max_body_size_middleware import MaxBodySizeMiddleware
from .pipeline_factory import build_pipeline
from .redis_pool import get_redis_pool

logger = logging.getLogger(__name__)

app = FastAPI(
    title="KTP Identification API",
    version="0.1.0",
    description="Detects an Indonesian KTP card in a photo and extracts its fields via OCR.",
)

# Rejects an oversized request body at the ASGI layer, before FastAPI's own
# multipart parsing (which applies no size cap to file parts, and runs
# before every Depends()-based check and the rate limiter below) ever reads
# it — see max_body_size_middleware.py for why enforcing this inside the
# endpoint, after the body is already fully buffered, is too late.
app.add_middleware(MaxBodySizeMiddleware, max_body_size=settings.max_upload_bytes)

# In-memory storage (the library's default), not Redis-backed: only the API
# process ever handles HTTP requests (the worker doesn't), and this project
# isn't horizontally scaled, so per-process state is correct here, not a
# shortcut — Redis-backed storage would be solving a scaling problem this
# deployment doesn't have. Revisit if the API is ever run as multiple
# replicas behind a load balancer.
#
# default_limits (applied via SlowAPIMiddleware below) rather than a
# per-route @limiter.limit(...) decorator: a decorated route is resolved by
# calling the endpoint function, which FastAPI only does *after*
# `dependencies=[Depends(require_api_key)]` has already been solved — so a
# request that fails auth never reaches the decorator's check at all, and
# was previously exempt from rate limiting entirely (confirmed empirically:
# repeated wrong-key requests never tripped the limiter). Slowapi's
# middleware runs before FastAPI's routing/dependency resolution, closing
# that gap; routes that shouldn't share this budget opt out via
# @limiter.exempt instead of opting in via a decorator.
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_STATIC_DIR = Path(__file__).parent / "static"

# Redis stores a queued job's arguments — here, the raw uploaded KTP photo
# bytes — until the job starts or this window elapses (arq's `_expires`).
# Left unset, arq defaults this to 24 hours, sized for generic job
# scheduling, not for how long sensitive PII should sit in Redis. Real
# processing finishes in seconds; 10 minutes gives comfortable headroom for
# retries (arq's own defaults: up to 5 tries, 300s timeout each) without
# leaving the raw photo around anywhere near a full day.
JOB_EXPIRES_SECONDS = 10 * 60


@app.get("/demo")
@limiter.exempt
async def demo() -> FileResponse:
    """A plain upload page for manual testing — drag/drop a photo, see the
    extracted fields rendered instead of raw JSON. Not authenticated (same
    as the rest of this dev-mode API); don't expose this beyond localhost.
    """
    return FileResponse(_STATIC_DIR / "demo.html")

# Used by the sync endpoint only. The async endpoints enqueue work onto
# Redis instead — see app/worker.py for the process that actually runs the
# pipeline for those.
pipeline = build_pipeline()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return  # auth disabled (dev only)
    # hmac.compare_digest rather than `!=`: a plain string comparison
    # short-circuits on the first differing byte, which is the classic
    # shape of a timing side-channel against a shared secret. This is a
    # free fix regardless of how practically exploitable that is here.
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
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
@limiter.exempt
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/ktp/extract", dependencies=[Depends(require_api_key)])
async def extract(file: UploadFile):
    contents = await _read_and_validate_upload(file)
    # pipeline.run() is synchronous and does real CPU/network work (model
    # inference, an HTTP call to the OCR service) — run it off the event
    # loop so one request's multi-second processing doesn't block every
    # other request this process is serving, including health checks.
    result = await asyncio.to_thread(pipeline.run, contents)

    status_code = 200
    if result.status == ExtractionStatus.INVALID_IMAGE:
        status_code = 422
    elif result.status == ExtractionStatus.NO_CARD_DETECTED:
        status_code = 404

    return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))


@app.post("/v1/ktp/jobs", dependencies=[Depends(require_api_key)])
async def create_job(file: UploadFile, redis=Depends(get_redis_pool)):
    contents = await _read_and_validate_upload(file)
    job = await redis.enqueue_job(
        "process_ktp_extraction", contents, _expires=JOB_EXPIRES_SECONDS
    )
    return {"job_id": job.job_id, "status": JobStatus.queued.value}


@app.get("/v1/ktp/jobs/{job_id}", dependencies=[Depends(require_api_key)])
@limiter.exempt
async def get_job(job_id: str, redis=Depends(get_redis_pool)):
    job = Job(job_id, redis)
    status = await job.status()
    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": status.value}
    if status == JobStatus.complete:
        try:
            response["result"] = await job.result(timeout=5)
        except Exception:  # noqa: BLE001 - surface any pipeline failure on the job
            # Log the real exception server-side only — the raw message can
            # contain internal details (paths, library internals) that
            # shouldn't go back to whoever is polling this job.
            logger.exception("Job %s failed during processing", job_id)
            response["status"] = "failed"
            response["error"] = "Job processing failed. Check server logs for details."
    return response
