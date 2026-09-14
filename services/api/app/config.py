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


settings = Settings()
