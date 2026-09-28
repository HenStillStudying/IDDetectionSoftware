"""Contract for locating and rectifying the KTP card within an arbitrary
input photo. Lives in libs so both the API service (the consumer) and the
detection service (the implementer) can depend on it without either one
depending on the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL.Image import Image as PILImage

from ktp_schema import BoundingBox


class MultipleCardsDetectedError(Exception):
    """More than one card was found in the image. Implementations raise
    this rather than picking one: with two IDs in view there's no telling
    whose identity the caller meant, and a silent pick would return
    someone's data as if it were unambiguous."""

    def __init__(self, count: int):
        super().__init__(f"{count} cards detected")
        self.count = count


class DetectionService(ABC):
    @abstractmethod
    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        """Returns (flattened upright card image, bbox in original image coords).

        Returns None if no card was found above the detector's confidence
        threshold. Raises MultipleCardsDetectedError if more than one was.
        """
