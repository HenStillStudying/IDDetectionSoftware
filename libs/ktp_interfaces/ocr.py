"""Contract for reading structured fields off an already-rectified, upright
card image. Lives in libs for the same reason as DetectionService.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL.Image import Image as PILImage

from ktp_schema import KtpFields


class OcrService(ABC):
    @abstractmethod
    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        """Returns every KTP field with a per-field confidence score."""
