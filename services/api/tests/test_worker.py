import asyncio
import io

from PIL import Image

from app.pipeline import KtpExtractionPipeline
from app.stub_models import StubDetectionService, StubOcrService
from app.worker import process_ktp_extraction


def _fake_photo_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_process_ktp_extraction_returns_json_serializable_result():
    ctx = {
        "pipeline": KtpExtractionPipeline(
            detection_service=StubDetectionService(),
            ocr_service=StubOcrService(),
        )
    }

    result = asyncio.run(process_ktp_extraction(ctx, _fake_photo_bytes()))

    assert isinstance(result, dict)
    assert result["status"] == "low_confidence"  # stub models always report 0 confidence
    assert "fields" in result
