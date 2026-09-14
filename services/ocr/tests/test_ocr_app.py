import io

from fastapi.testclient import TestClient
from PIL import Image

from ocr_app.main import app, get_ocr_service
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


app.dependency_overrides[get_ocr_service] = lambda: FakeOcrService()
client = TestClient(app)


def _fake_photo_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_returns_fields_from_the_configured_ocr_service():
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("card.jpg", _fake_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["nik"]["value"] == "FAKE"
    assert body["nik"]["confidence"] == 0.9


def test_extract_rejects_invalid_image():
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422


def test_extract_rejects_empty_file():
    resp = client.post(
        "/v1/ocr/extract",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400
