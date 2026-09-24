import io

from PIL import Image
from PIL.Image import Image as PILImage

from app.pipeline import MAX_UPLOAD_DIMENSION, KtpExtractionPipeline
from app.stub_models import StubOcrService
from ktp_interfaces import DetectionService, OcrService
from ktp_schema import BoundingBox, ExtractionStatus, FieldValue, KtpFields


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


# --- NIK consistency check, through the pipeline ---
# 3205121507900007 encodes 15-07-(19)90, male.
_MALE_NIK = "3205121507900007"


class _FixedOcrService(OcrService):
    """Returns chosen values/confidences for the fields the consistency
    check reads; everything else empty."""

    def __init__(self, nik, tanggal_lahir, jenis_kelamin, confidence=0.97, date_confidence=None):
        self._nik = FieldValue(value=nik, confidence=confidence)
        self._date = FieldValue(
            value=tanggal_lahir, confidence=confidence if date_confidence is None else date_confidence
        )
        self._gender = FieldValue(value=jenis_kelamin, confidence=confidence)

    def extract_fields(self, rectified_card) -> KtpFields:
        empty = FieldValue(value=None, confidence=0.0)
        fields = {name: empty for name in KtpFields.model_fields}
        fields.update(nik=self._nik, tanggal_lahir=self._date, jenis_kelamin=self._gender)
        return KtpFields(**fields)


def _run(ocr: OcrService):
    return KtpExtractionPipeline(_RecordingDetectionService(), ocr).run(_photo_bytes((200, 120)))


def test_self_contradicting_card_is_flagged():
    result = _run(_FixedOcrService(_MALE_NIK, "11-12-1989", "PEREMPUAN"))
    assert result.nik_consistency is not None
    assert result.nik_consistency.consistent is False
    assert len(result.nik_consistency.mismatches) == 2
    assert sum("NIK consistency" in w for w in result.warnings) == 2


def test_self_consistent_card_passes():
    result = _run(_FixedOcrService(_MALE_NIK, "15-07-1990", "LAKI-LAKI"))
    assert result.nik_consistency.consistent is True
    assert result.nik_consistency.fields_checked == ["tanggal_lahir", "jenis_kelamin"]
    assert not any("NIK consistency" in w for w in result.warnings)


def test_low_confidence_field_is_left_out_of_the_comparison():
    # A shaky birth-date read (possibly an OCR misread) must not be
    # compared — only the confidently read gender is.
    result = _run(_FixedOcrService(_MALE_NIK, "11-12-1989", "LAKI-LAKI", date_confidence=0.5))
    assert result.nik_consistency.fields_checked == ["jenis_kelamin"]
    assert result.nik_consistency.consistent is True


def test_low_confidence_nik_skips_the_check_entirely():
    # One misread NIK digit would look exactly like a contradiction.
    result = _run(_FixedOcrService(_MALE_NIK, "11-12-1989", "PEREMPUAN", confidence=0.5))
    assert result.nik_consistency is None
