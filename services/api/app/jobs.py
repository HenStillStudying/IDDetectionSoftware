"""In-memory async job store.

This previews the async API contract (submit → poll) for batch callers
without standing up real queue infrastructure yet. It is intentionally not
production-grade: state is lost on restart and doesn't scale past one
process. Swap this for a real queue (SQS/RabbitMQ) + persistent job table
backed by Redis/Postgres before load or multi-instance deployment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from ktp_schema import KtpExtractionResult


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    result: KtpExtractionResult | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_processing(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = JobStatus.PROCESSING

    def mark_done(self, job_id: str, result: KtpExtractionResult) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.DONE
            job.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
