"""OcrService implementation that calls the separately-deployed OCR
microservice (services/ocr) over HTTP, rather than running PaddleOCR in
this process.

This is what actually unblocks GPU-accelerated OCR: PaddleOCR's CUDA/cuDNN
DLLs collide with PyTorch's (used by YoloDetectionService) when both are
loaded in the same Windows process. Keeping them in separate processes
sidesteps that, and lets this GPU-bound stage scale independently of the
cheap detection stage.
"""

from __future__ import annotations

import io

import httpx
from PIL.Image import Image as PILImage

from ktp_interfaces import OcrService
from ktp_schema import KtpFields

DEFAULT_TIMEOUT_SECONDS = 30.0


class RemoteOcrService(OcrService):
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        internal_key: str | None = None,
    ):
        headers = {"X-Internal-Key": internal_key} if internal_key else None
        self._client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        buffer = io.BytesIO()
        rectified_card.save(buffer, format="JPEG")
        buffer.seek(0)

        response = self._client.post(
            "/v1/ocr/extract",
            files={"file": ("card.jpg", buffer, "image/jpeg")},
        )
        response.raise_for_status()
        return KtpFields.model_validate(response.json())
