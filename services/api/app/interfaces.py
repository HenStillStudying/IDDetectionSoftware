"""Abstract contracts for the two model-backed stages of the pipeline.

The API layer is written against these interfaces so the detection model
(services/detection) and OCR model (services/ocr) can be developed and
swapped independently — the API doesn't need to know whether inference
happens in-process, via a local ONNX runtime, or over gRPC to a separate
model-serving container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL.Image import Image as PILImage

from ktp_schema import BoundingBox, KtpFields


class DetectionService(ABC):
    """Locates and rectifies the KTP card within an arbitrary input photo."""

    @abstractmethod
    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        """Returns (flattened upright card image, bbox in original image coords).

        Returns None if no card was found above the detector's confidence
        threshold.
        """


class OcrService(ABC):
    """Reads structured fields off an already-rectified, upright card image."""

    @abstractmethod
    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        """Returns every KTP field with a per-field confidence score."""
