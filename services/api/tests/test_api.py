import io
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
from arq.connections import ArqRedis
from arq.jobs import JobStatus
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import KtpExtractionPipeline
from app.redis_pool import get_redis_pool
from app.stub_models import StubOcrService
from ktp_interfaces import DetectionService
from ktp_schema import BoundingBox


@pytest.fixture
def client():
    # A fresh fake pool per test, and `with TestClient(...)` (not a bare
    # instance) so every request in a test shares one event loop — fakeredis's
    # async internals bind to the first event loop that touches them, and
    # TestClient otherwise runs each call in its own loop, which breaks a
    # pool reused across calls.
    fake_pool = ArqRedis(connection_pool=fakeredis.FakeAsyncRedis().connection_pool)
    app.dependency_overrides[get_redis_pool] = lambda: fake_pool
    # The rate limiter's in-memory counters live on the module-level `app`
    # object, so without resetting them here, one test hitting the limit
    # would silently poison every later test sharing the same test-client
    # source IP.
    app.state.limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_redis_pool, None)


def _fake_photo_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_demo_page_is_served(client):
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert b"<title>KTP Identification" in resp.content


def test_extract_with_stub_pipeline_returns_low_confidence(client):
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "low_confidence"  # stub models always report 0 confidence
    assert body["bounding_box"] is not None
    assert body["fields"] is not None


def test_extract_rejects_invalid_image(client):
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422
    assert resp.json()["status"] == "invalid_image"


def test_extract_rejects_empty_file(client):
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_job_lifecycle_enqueues_and_can_be_polled(client):
    # No worker is running in this test, so this exercises the enqueue +
    # poll contract (job accepted, appears in Redis, pollable by id) —
    # actual pipeline execution is the worker's job, tested separately
    # against app.worker.process_ktp_extraction.
    create_resp = client.post(
        "/v1/ktp/jobs",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert create_resp.status_code == 200
    body = create_resp.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]

    poll_resp = client.get(f"/v1/ktp/jobs/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] in {"queued", "deferred", "in_progress"}


def test_job_not_found(client):
    resp = client.get("/v1/ktp/jobs/does-not-exist")
    assert resp.status_code == 404


def test_job_failure_error_message_is_sanitized(client):
    # A failed job's real exception (which can contain internal paths or
    # other implementation details) must never reach the client directly —
    # only a generic message, with the real one logged server-side instead.
    real_error = RuntimeError("/internal/secret/path leaked, api_key=xyz123")
    with (
        patch("app.main.Job.status", new=AsyncMock(return_value=JobStatus.complete)),
        patch("app.main.Job.result", new=AsyncMock(side_effect=real_error)),
    ):
        resp = client.get("/v1/ktp/jobs/some-id")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "Job processing failed. Check server logs for details."
    assert "secret" not in body["error"]
    assert "xyz123" not in body["error"]


def test_extract_is_rate_limited_after_threshold(client):
    # The limit is 15/minute; the 16th request from the same (test-client)
    # source IP within that window should be rejected rather than doing
    # real work.
    for _ in range(15):
        resp = client.post(
            "/v1/ktp/extract",
            files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 429


def test_create_job_is_rate_limited_after_threshold(client):
    for _ in range(15):
        resp = client.post(
            "/v1/ktp/jobs",
            files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    resp = client.post(
        "/v1/ktp/jobs",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 429


def test_job_polling_is_not_rate_limited(client):
    # Polling an existing job is a cheap Redis lookup, not the expensive
    # work the limit exists to protect — a client checking its own job's
    # status repeatedly shouldn't share that budget.
    for _ in range(20):
        resp = client.get("/v1/ktp/jobs/does-not-exist")
        assert resp.status_code == 404  # never 429


class _SlowDetectionService(DetectionService):
    """Sleeps synchronously like a real CPU-bound model call would — used
    to prove /v1/ktp/extract doesn't block the event loop while it runs,
    not just that its response is unchanged.
    """

    def detect_and_rectify(self, image):
        time.sleep(0.3)
        width, height = image.size
        return image, BoundingBox(x1=0, y1=0, x2=width, y2=height, detection_confidence=1.0)


def test_extract_does_not_block_the_event_loop(client):
    # Regression test for a real bug: pipeline.run() used to be called
    # directly inside the async handler with no threading, which blocked
    # the whole process (including every other concurrent request) for the
    # full duration of each request. Two concurrent 0.3s requests should
    # overlap and finish in well under 2x0.3s if the fix holds; before the
    # fix, they'd serialize to ~0.6s+.
    slow_pipeline = KtpExtractionPipeline(_SlowDetectionService(), StubOcrService())
    with patch("app.main.pipeline", slow_pipeline):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    client.post,
                    "/v1/ktp/extract",
                    files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
                )
                for _ in range(2)
            ]
            responses = [f.result() for f in futures]
        elapsed = time.perf_counter() - started

    for resp in responses:
        assert resp.status_code == 200
    assert elapsed < 0.5
