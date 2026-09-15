"""Builds the same KtpExtractionPipeline both the API process and the async
worker use, from the same env-var-driven config, so the two never drift
into different wiring by accident.
"""

from __future__ import annotations

import logging
import os

from ktp_interfaces import DetectionService, OcrService

from .config import settings
from .pipeline import KtpExtractionPipeline
from .remote_ocr_service import RemoteOcrService
from .stub_models import StubDetectionService, StubOcrService

logger = logging.getLogger(__name__)


def build_detection_service() -> DetectionService:
    """Uses the trained YOLO detector when KTP_DETECTION_WEIGHTS points at a
    real weights file; falls back to the stub (0 confidence, whole image as
    the card) otherwise, so the service still boots in dev/CI without a
    trained model.
    """
    weights_path = settings.detection_weights_path
    if weights_path and os.path.exists(weights_path):
        from ktp_detection import YoloDetectionService

        logger.info("Loading YOLO detection model from %s", weights_path)
        return YoloDetectionService(weights_path)

    logger.warning("KTP_DETECTION_WEIGHTS not set or missing — using stub detection service")
    return StubDetectionService()


def build_ocr_service() -> OcrService:
    """Calls the separately-deployed OCR microservice (services/ocr) over
    HTTP when KTP_OCR_SERVICE_URL is set; falls back to the stub otherwise,
    so the service still boots without that dependency running.

    OCR runs as its own service (not imported in-process here) because
    PaddleOCR's GPU build crashes with Windows DLL conflicts when loaded
    alongside PyTorch (used by the detection service below) — see
    services/ocr's README for the full story.
    """
    if settings.ocr_service_url:
        logger.info("Using remote OCR service at %s", settings.ocr_service_url)
        return RemoteOcrService(settings.ocr_service_url)

    logger.warning("KTP_OCR_SERVICE_URL not set — using stub OCR service")
    return StubOcrService()


def build_pipeline() -> KtpExtractionPipeline:
    return KtpExtractionPipeline(
        detection_service=build_detection_service(),
        ocr_service=build_ocr_service(),
    )
