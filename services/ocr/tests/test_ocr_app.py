import io
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_app.main import MAX_UPLOAD_BYTES, app, get_ocr_service
from ktp_schema import FieldValue, KtpFields


class FakeOcrService:
    def extract_fields(self, rectified_card) -> KtpFields:
        empty = FieldValue(value=None, confidence=0.0)
        filled = FieldValue(value="FAKE", confidence=0.9)
        return KtpFields(
            nik=filled,
            nama=filled,
            tempat_lahir=empty,
            tanggal_lahir=empty,
            jenis_kelamin=empty,
            golongan_darah=empty,
            alamat=empty,
            rt_rw=empty,
            kelurahan_desa=empty,
            kecamatan=empty,
            agama=empty,
            status_perkawinan=empty,
            pekerjaan=empty,
            kewarganegaraan=empty,
            berlaku_hingga=empty,
            provinsi=empty,
            kota_kabupaten=empty,
        )


@pytest.fixture
def client():
    # `with TestClient(...)` (not a bare instance) so every request in a
    # test shares one persistent event loop/portal — without it, each call
    # appears to get its own isolated loop, which would make a concurrency
    # test like test_extract_does_not_block_the_event_loop pass regardless
    # of whether the handler actually blocks that loop (confirmed: a plain
    # `client = TestClient(app)` let that test pass even with the bug it's
    # meant to catch still present).
    app.dependency_overrides[get_ocr_service] = lambda: FakeOcrService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_ocr_service, None)


def _fake_photo_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_ignores_internal_key_when_not_configured(client):
    # Default/dev state: KTP_OCR_INTERNAL_KEY unset, so no header is
    # required at all — matches every other unauthenticated route today.
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200


def test_extract_rejects_missing_or_wrong_internal_key_when_configured(client):
    with patch("ocr_app.main._INTERNAL_KEY", "the-real-key"):
        resp = client.post(
            "/v1/ocr/extract",
            files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 401

        resp = client.post(
            "/v1/ocr/extract",
            files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
            headers={"X-Internal-Key": "wrong-key"},
        )
        assert resp.status_code == 401


def test_extract_accepts_correct_internal_key_when_configured(client):
    with patch("ocr_app.main._INTERNAL_KEY", "the-real-key"):
        resp = client.post(
            "/v1/ocr/extract",
            files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
            headers={"X-Internal-Key": "the-real-key"},
        )
        assert resp.status_code == 200


def test_extract_returns_fields_from_the_configured_ocr_service(client):
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["nik"]["value"] == "FAKE"
    assert body["nik"]["confidence"] == 0.9


def test_extract_rejects_invalid_image(client):
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422


def test_extract_rejects_decompression_bomb_cleanly(client, monkeypatch):
    # Regression test for a real bug: this handler's except tuple was
    # missing Image.DecompressionBombError (a direct Exception subclass,
    # unrelated to UnidentifiedImageError/OSError), so a maliciously huge
    # image produced an unhandled 500 instead of the same clean 422 any
    # other bad upload gets — the exact bug already fixed once in
    # services/api/app/pipeline.py. Lowering MAX_IMAGE_PIXELS (rather than
    # constructing an actual huge image) keeps this test fast.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("huge.jpg", _fake_photo_bytes(), "image/jpeg")},  # 200x120 = 24,000px > 100
    )
    assert resp.status_code == 422


def test_extract_rejects_empty_file(client):
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


class _SlowFakeOcrService(FakeOcrService):
    """Sleeps synchronously like a real CPU-bound OCR call would — used to
    prove /v1/ocr/extract doesn't block the event loop while it runs, not
    just that its response is unchanged.
    """

    def extract_fields(self, rectified_card) -> KtpFields:
        time.sleep(0.3)
        return super().extract_fields(rectified_card)


def test_extract_does_not_block_the_event_loop(client):
    # Regression test for a real bug: extract_fields() used to be called
    # directly inside the async handler with no threading, which blocked
    # the whole process (including every other concurrent request) for the
    # full duration of each request. Two concurrent 0.3s requests should
    # overlap and finish in well under 2x0.3s if the fix holds; before the
    # fix, they'd serialize to ~0.6s+.
    app.dependency_overrides[get_ocr_service] = lambda: _SlowFakeOcrService()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                client.post,
                "/v1/ocr/extract",
                files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
            )
            for _ in range(2)
        ]
        responses = [f.result() for f in futures]
    elapsed = time.perf_counter() - started

    for resp in responses:
        assert resp.status_code == 200
    assert elapsed < 0.5


def test_extract_rejects_oversized_upload_before_reading_it(client):
    # Regression test for a real bug: this endpoint's own size check ran
    # only after `await file.read()` had already fully buffered the request
    # body — a bigger problem here than on the api service since this
    # service has no auth of its own at all. Proves MaxBodySizeMiddleware
    # now rejects it at the ASGI layer before extract_fields ever runs.
    spy_service = FakeOcrService()
    spy_service.extract_fields = MagicMock(wraps=spy_service.extract_fields)
    app.dependency_overrides[get_ocr_service] = lambda: spy_service

    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413
    # The middleware and the endpoint's own (now-redundant-in-this-case)
    # size check return distinctly worded messages — asserting on the
    # middleware's wording is what actually proves the ASGI layer caught
    # this, not just that a 413 came back from somewhere.
    assert "Request body exceeds the maximum allowed size" in resp.json()["detail"]
    spy_service.extract_fields.assert_not_called()
