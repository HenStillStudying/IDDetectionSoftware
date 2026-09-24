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
    NikConsistency,
    check_nik_consistency,
    validate_nik,
)
from ktp_schema.result import NikValidation
from ktp_interfaces import DetectionService, OcrService

from . import forensics

LOW_CONFIDENCE_THRESHOLD = 0.6

# Fields only take part in the NIK consistency check when read at least
# this confidently — including the NIK itself. A single OCR misread digit
# would otherwise look exactly like a self-contradicting (fabricated) card.
# Correct reads in this project typically score ~0.96-0.99.
CONSISTENCY_MIN_CONFIDENCE = 0.8

# A real phone photo runs several times larger than the 960x720 synthetic
# images this project was benchmarked against — measured directly, that
# gap alone took a real request from ~4s to ~13s, since both YOLO detection
# and PaddleOCR's stages (running on the detected crop) scale with image
# area. 1600px keeps the card's smallest legible text (a 16-digit NIK)
# comfortably readable after the crop while cutting a typical 3000-4000px
# phone photo's pixel count by roughly 4-6x. Only ever shrinks — an
# already-small upload is untouched.
MAX_UPLOAD_DIMENSION = 1600


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
            image = _downscale_if_needed(image)
        except (UnidentifiedImageError, OSError):
            return KtpExtractionResult(
                status=ExtractionStatus.INVALID_IMAGE,
                warnings=["Could not decode the uploaded file as an image."],
                processing_time_ms=_elapsed_ms(started),
            )
        except Image.DecompressionBombError:
            # Pillow's own default limit (Image.MAX_IMAGE_PIXELS) already
            # prevents the actual memory-exhaustion risk here — this is
            # just making sure a maliciously huge image gets the same
            # clean validation response as any other bad upload, not an
            # unhandled 500.
            return KtpExtractionResult(
                status=ExtractionStatus.INVALID_IMAGE,
                warnings=["Image exceeds the maximum allowed pixel count."],
                processing_time_ms=_elapsed_ms(started),
            )

        # Tier-1 tamper-detection heuristics (EXIF inspection only for now —
        # see forensics.py) — run on the original bytes, since our own
        # resize/re-encoding could strip the EXIF this checks. Runs
        # regardless of whether a card is even found below: it's a property
        # of the uploaded photo, not of a successful extraction.
        forensics_result = forensics.analyze(image_bytes)

        detection = self._detection_service.detect_and_rectify(image)
        if detection is None:
            return KtpExtractionResult(
                status=ExtractionStatus.NO_CARD_DETECTED,
                forensics=forensics_result,
                warnings=["No KTP card was detected in the image."] + forensics_result.warnings,
                processing_time_ms=_elapsed_ms(started),
            )

        rectified_card, bbox = detection
        fields = self._ocr_service.extract_fields(rectified_card)

        nik_validation = self._validate_nik_field(fields.nik.value)
        overall_confidence = self._overall_confidence(bbox, fields)

        nik_consistency = self._check_nik_consistency(fields, nik_validation)

        warnings: list[str] = list(forensics_result.warnings)
        if nik_validation is not None and not nik_validation.is_valid:
            warnings.extend(f"NIK: {err}" for err in nik_validation.errors)
        if nik_consistency is not None and not nik_consistency.consistent:
            warnings.extend(f"NIK consistency: {m}" for m in nik_consistency.mismatches)

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
            nik_consistency=nik_consistency,
            forensics=forensics_result,
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
    def _check_nik_consistency(
        fields: KtpFields, nik_validation: NikValidation | None
    ) -> NikConsistency | None:
        if nik_validation is None or not nik_validation.is_valid:
            return None
        if fields.nik.confidence < CONSISTENCY_MIN_CONFIDENCE:
            return None

        def confident(field) -> str | None:
            return field.value if field.confidence >= CONSISTENCY_MIN_CONFIDENCE else None

        checked, mismatches = check_nik_consistency(
            fields.nik.value, confident(fields.tanggal_lahir), confident(fields.jenis_kelamin)
        )
        if not checked:
            return None
        return NikConsistency(fields_checked=checked, consistent=not mismatches, mismatches=mismatches)

    @staticmethod
    def _overall_confidence(bbox: BoundingBox, fields: KtpFields) -> float:
        field_confidences = [
            field_value.confidence for _, field_value in fields
        ]
        all_confidences = [bbox.detection_confidence, *field_confidences]
        return sum(all_confidences) / len(all_confidences)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _downscale_if_needed(image: Image.Image, max_dimension: int = MAX_UPLOAD_DIMENSION) -> Image.Image:
    """Shrinks an oversized upload to at most `max_dimension` on its longer
    side, preserving aspect ratio. A no-op when the image is already within
    that bound, so a small upload never gets needlessly re-encoded.
    """
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_dimension:
        return image

    scale = max_dimension / longest_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.LANCZOS)
