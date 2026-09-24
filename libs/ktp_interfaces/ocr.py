"""Contract for reading structured fields off an already-rectified, upright
card image. Lives in libs for the same reason as DetectionService.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL.Image import Image as PILImage

from ktp_schema import KtpFields


class OcrUnavailableError(Exception):
    """The OCR backend couldn't produce a result: unreachable, erroring, or
    answering with something that isn't valid KTP fields. Implementations
    raise this instead of their transport's own exceptions, so callers can
    report it cleanly without knowing how OCR is reached."""


class OcrTimeoutError(OcrUnavailableError):
    """The OCR backend didn't answer in time."""


class OcrService(ABC):
    @abstractmethod
    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        """Returns every KTP field with a per-field confidence score.

        Raises OcrUnavailableError (or OcrTimeoutError) when the backend
        can't produce a result.
        """
