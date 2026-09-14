import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _fake_photo_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_with_stub_pipeline_returns_low_confidence():
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "low_confidence"  # stub models always report 0 confidence
    assert body["bounding_box"] is not None
    assert body["fields"] is not None


def test_extract_rejects_invalid_image():
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422
    assert resp.json()["status"] == "invalid_image"


def test_extract_rejects_empty_file():
    resp = client.post(
        "/v1/ktp/extract",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_job_lifecycle():
    create_resp = client.post(
        "/v1/ktp/jobs",
        files={"file": ("photo.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    poll_resp = client.get(f"/v1/ktp/jobs/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] in {"pending", "processing", "done"}


def test_job_not_found():
    resp = client.get("/v1/ktp/jobs/does-not-exist")
    assert resp.status_code == 404
