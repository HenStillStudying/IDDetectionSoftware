"""Orchestrates a single extraction request through detection, OCR, and
field validation, and settles on one overall status/confidence for the
response contract.
"""

from __future__ import annotations

import io
import time

from PIL import Image, UnidentifiedImageError

from ktp_schema import (
    BoundingBox,
    ExtractionStatus,
    KtpExtractionResult,
    KtpFields,
    validate_nik,
)
from ktp_schema.result import NikValidation

from .interfaces import DetectionService, OcrService

LOW_CONFIDENCE_THRESHOLD = 0.6


class KtpExtractionPipeline:
    def __init__(self, detection_service: DetectionService, ocr_service: OcrService):
        self._detection_service = detection_service
        self._ocr_service = ocr_service

    def run(self, image_bytes: bytes) -> KtpExtractionResult:
        started = time.perf_counter()

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError):
            return KtpExtractionResult(
                status=ExtractionStatus.INVALID_IMAGE,
                warnings=["Could not decode the uploaded file as an image."],
                processing_time_ms=_elapsed_ms(started),
            )

        detection = self._detection_service.detect_and_rectify(image)
        if detection is None:
            return KtpExtractionResult(
                status=ExtractionStatus.NO_CARD_DETECTED,
                warnings=["No KTP card was detected in the image."],
                processing_time_ms=_elapsed_ms(started),
            )

        rectified_card, bbox = detection
        fields = self._ocr_service.extract_fields(rectified_card)

        nik_validation = self._validate_nik_field(fields.nik.value)
        overall_confidence = self._overall_confidence(bbox, fields)

        warnings: list[str] = []
        if nik_validation is not None and not nik_validation.is_valid:
            warnings.extend(f"NIK: {err}" for err in nik_validation.errors)

        status = (
            ExtractionStatus.LOW_CONFIDENCE
            if overall_confidence < LOW_CONFIDENCE_THRESHOLD
            else ExtractionStatus.OK
        )

        return KtpExtractionResult(
            status=status,
            bounding_box=bbox,
            fields=fields,
            nik_validation=nik_validation,
            overall_confidence=overall_confidence,
            warnings=warnings,
            processing_time_ms=_elapsed_ms(started),
        )

    @staticmethod
    def _validate_nik_field(nik_value: str | None) -> NikValidation | None:
        if not nik_value:
            return None
        result = validate_nik(nik_value)
        return NikValidation(is_valid=result.is_valid, errors=result.errors)

    @staticmethod
    def _overall_confidence(bbox: BoundingBox, fields: KtpFields) -> float:
        field_confidences = [
            field_value.confidence for _, field_value in fields
        ]
        all_confidences = [bbox.detection_confidence, *field_confidences]
        return sum(all_confidences) / len(all_confidences)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
