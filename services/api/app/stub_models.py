"""Placeholder implementations of DetectionService/OcrService.

These exist so the API's request/response contract, error handling, and job
plumbing can be built and tested before the real YOLO detector and OCR model
are trained. Nothing here should be trusted for real card data — swap these
out (see services/detection and services/ocr) before going anywhere near
production traffic.
"""

from __future__ import annotations

from PIL.Image import Image as PILImage

from ktp_interfaces import DetectionService, OcrService
from ktp_schema import BoundingBox, FieldValue, KtpFields


class StubDetectionService(DetectionService):
    """Treats the entire input image as the card. No real detection."""

    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        width, height = image.size
        bbox = BoundingBox(x1=0, y1=0, x2=width, y2=height, detection_confidence=0.0)
        return image, bbox


class StubOcrService(OcrService):
    """Returns every field empty with zero confidence. No real OCR."""

    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        empty = FieldValue(value=None, confidence=0.0)
        return KtpFields(
            nik=empty,
            nama=empty,
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
