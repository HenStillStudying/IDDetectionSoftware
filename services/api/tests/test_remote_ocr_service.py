from app.remote_ocr_service import RemoteOcrService


def test_sends_internal_key_header_when_configured():
    service = RemoteOcrService("http://127.0.0.1:8001", internal_key="secret-value")
    assert service._client.headers["x-internal-key"] == "secret-value"


def test_omits_internal_key_header_when_not_configured():
    service = RemoteOcrService("http://127.0.0.1:8001")
    assert "x-internal-key" not in service._client.headers
