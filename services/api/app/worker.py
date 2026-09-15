"""arq worker: consumes KTP extraction jobs from Redis.

Run with:
    arq app.worker.WorkerSettings

This is the production-grade replacement for the old in-process
BackgroundTasks + in-memory JobStore — jobs now survive an API restart
(they live in Redis, not this process's memory) and processing happens in
a separate process that can be scaled independently of the API.

arq is async-native rather than fork-based (unlike RQ), which matters here
because this runs on Windows during dev — os.fork() doesn't exist there.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from arq.connections import RedisSettings

from .config import settings
from .pipeline_factory import build_pipeline

logger = logging.getLogger(__name__)


async def on_startup(ctx: dict[str, Any]) -> None:
    # Built once per worker process, not per job — the detection model and
    # OCR HTTP client are expensive enough to set up that doing it per job
    # would dominate the actual work.
    logger.info("Worker starting up: building extraction pipeline")
    ctx["pipeline"] = build_pipeline()


async def process_ktp_extraction(ctx: dict[str, Any], image_bytes: bytes) -> dict:
    # pipeline.run() is synchronous and does real CPU/network work (model
    # inference, an HTTP call to the OCR service) — run it off the event
    # loop so it doesn't block arq's own housekeeping (health checks, job
    # polling) for the whole duration.
    result = await asyncio.to_thread(ctx["pipeline"].run, image_bytes)
    return result.model_dump(mode="json")


class WorkerSettings:
    functions = [process_ktp_extraction]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
