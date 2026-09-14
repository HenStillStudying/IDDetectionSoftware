"""Contract for locating and rectifying the KTP card within an arbitrary
input photo. Lives in libs so both the API service (the consumer) and the
detection service (the implementer) can depend on it without either one
depending on the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL.Image import Image as PILImage

from ktp_schema import BoundingBox


class DetectionService(ABC):
    @abstractmethod
    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        """Returns (flattened upright card image, bbox in original image coords).

        Returns None if no card was found above the detector's confidence
        threshold.
        """
