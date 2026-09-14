"""The API-facing response contract for a KTP extraction request."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .fields import KtpFields


class ExtractionStatus(str, Enum):
    OK = "ok"
    NO_CARD_DETECTED = "no_card_detected"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_IMAGE = "invalid_image"
    ERROR = "error"


class BoundingBox(BaseModel):
    """Pixel coordinates of the detected card in the original input image."""

    x1: int
    y1: int
    x2: int
    y2: int
    detection_confidence: float = Field(ge=0.0, le=1.0)


class NikValidation(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class KtpExtractionResult(BaseModel):
    status: ExtractionStatus
    bounding_box: BoundingBox | None = None
    fields: KtpFields | None = None
    nik_validation: NikValidation | None = None
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None
