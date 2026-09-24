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


class NikConsistency(BaseModel):
    """Whether the NIK agrees with the birth date and gender printed on the
    same card. `consistent=False` means the card contradicts itself — a
    genuine card can't, so it's a strong sign of a fabricated or tampered
    card (or, rarely, an OCR misread that still cleared the confidence
    gate). `consistent=True` only means these two checks passed; it is not
    proof the card is genuine — anyone who knows the NIK format can make a
    fake agree with itself.
    """

    fields_checked: list[str]
    consistent: bool
    mismatches: list[str] = Field(default_factory=list)


class ForensicsResult(BaseModel):
    """Tier-1 tamper-detection signals on the uploaded photo — classical
    image-forensics heuristics, not a verdict. A clean result here is not
    proof of authenticity; `suspicious=True` is not proof of forgery — it's
    a signal meant to route a case for closer review, the same way a low
    OCR confidence flags a read without asserting it's wrong.

    Only EXIF inspection is active today — an Error Level Analysis check
    was tried and shelved after it false-positived on a real, unedited KTP
    photo; see services/api/app/forensics.py for why.
    """

    exif_present: bool
    editor_software_detected: str | None = None
    suspicious: bool
    warnings: list[str] = Field(default_factory=list)


class KtpExtractionResult(BaseModel):
    status: ExtractionStatus
    bounding_box: BoundingBox | None = None
    fields: KtpFields | None = None
    nik_validation: NikValidation | None = None
    # None when not checkable: no structurally valid NIK, or neither birth
    # date nor gender was read confidently enough to compare against it.
    nik_consistency: NikConsistency | None = None
    forensics: ForensicsResult | None = None
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None
