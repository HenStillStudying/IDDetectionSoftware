import io

import fakeredis
import pytest
from arq.connections import ArqRedis
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.redis_pool import get_redis_pool


@pytest.fixture
def client():
    # A fresh fake pool per test, and `with TestClient(...)` (not a bare
    # instance) so every request in a test shares one event loop — fakeredis's
    # async internals bind to the first event loop that touches them, and
    # TestClient otherwise runs each call in its own loop, which breaks a
    # pool reused across calls.
    fake_pool = ArqRedis(connection_pool=fakeredis.FakeAsyncRedis().connection_pool)
    app.dependency_overrides[get_redis_pool] = lambda: fake_pool
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
