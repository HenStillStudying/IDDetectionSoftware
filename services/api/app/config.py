"""Runtime configuration, read from environment variables.

Kept dependency-free (no pydantic-settings) since this is a handful of
scalar values; revisit if the settings surface grows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = int(os.getenv("KTP_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    api_key: str | None = os.getenv("KTP_API_KEY")  # None disables auth (dev only)
    detection_weights_path: str | None = os.getenv("KTP_DETECTION_WEIGHTS")
    ocr_service_url: str | None = os.getenv("KTP_OCR_SERVICE_URL")
    ocr_internal_key: str | None = os.getenv("KTP_OCR_INTERNAL_KEY")  # None disables (dev only)
    redis_url: str = os.getenv("KTP_REDIS_URL", "redis://localhost:6379")


settings = Settings()
