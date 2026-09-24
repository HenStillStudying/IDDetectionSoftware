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
from pydantic import ValidationError

from ktp_interfaces import OcrService, OcrTimeoutError, OcrUnavailableError
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

        # Translate httpx/pydantic failures into the OcrService contract's own
        # exceptions, so callers can report "OCR unavailable" cleanly instead
        # of an unhandled 500. The original exception is chained (`from`) for
        # server-side logs; it is never meant to reach an API client.
        try:
            response = self._client.post(
                "/v1/ocr/extract",
                files={"file": ("card.jpg", buffer, "image/jpeg")},
            )
            response.raise_for_status()
            return KtpFields.model_validate(response.json())
        except httpx.TimeoutException as exc:
            raise OcrTimeoutError("OCR service did not respond in time") from exc
        except httpx.HTTPStatusError as exc:
            # e.g. 401 = KTP_OCR_INTERNAL_KEY mismatch between the services.
            raise OcrUnavailableError(f"OCR service returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise OcrUnavailableError("OCR service unreachable") from exc
        except (ValueError, ValidationError) as exc:  # non-JSON body, or not KtpFields
            raise OcrUnavailableError("OCR service returned an invalid response") from exc
