import io

from PIL import Image
from PIL.Image import Image as PILImage

from app.pipeline import MAX_UPLOAD_DIMENSION, KtpExtractionPipeline
from app.stub_models import StubOcrService
from ktp_interfaces import DetectionService
from ktp_schema import BoundingBox, ExtractionStatus


class _RecordingDetectionService(DetectionService):
    """Records the image it was actually handed, so tests can assert on
    what the pipeline passed to detection without needing a real model.
    """

    def __init__(self) -> None:
        self.received_size: tuple[int, int] | None = None

    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        self.received_size = image.size
        width, height = image.size
        return image, BoundingBox(x1=0, y1=0, x2=width, y2=height, detection_confidence=1.0)


def _photo_bytes(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_oversized_upload_is_downscaled_before_detection():
    detection = _RecordingDetectionService()
    pipeline = KtpExtractionPipeline(detection, StubOcrService())

    pipeline.run(_photo_bytes((4000, 3000)))

    assert detection.received_size is not None
    assert max(detection.received_size) == MAX_UPLOAD_DIMENSION
    # Aspect ratio preserved (4000x3000 is 4:3).
    width, height = detection.received_size
    assert round(width / height, 2) == round(4000 / 3000, 2)


def test_small_upload_is_not_resized():
    detection = _RecordingDetectionService()
    pipeline = KtpExtractionPipeline(detection, StubOcrService())

    pipeline.run(_photo_bytes((200, 120)))

    assert detection.received_size == (200, 120)


def test_decompression_bomb_image_is_rejected_cleanly(monkeypatch):
    # Pillow's own default (Image.MAX_IMAGE_PIXELS) already prevents the
    # real memory-exhaustion risk; this only checks that hitting it returns
    # a clean validation response instead of an unhandled 500. Lowering the
    # limit here (rather than constructing an actual huge image) keeps the
    # test fast.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    detection = _RecordingDetectionService()
    pipeline = KtpExtractionPipeline(detection, StubOcrService())

    result = pipeline.run(_photo_bytes((200, 120)))  # 24,000 px > the lowered limit

    assert result.status == ExtractionStatus.INVALID_IMAGE
    assert detection.received_size is None  # never reached detection
