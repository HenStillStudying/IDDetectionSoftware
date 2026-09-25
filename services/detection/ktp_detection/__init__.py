import os

# Must run before ultralytics is first imported (it reads these at import).
# Importing ultralytics monkey-patches PIL.Image.open process-wide, and on
# *any* image that fails to open — so any bad or unusual upload — it ran
# `pip install pi-heif` from PyPI inside the running service: unpinned,
# triggered by user input. It also sends usage telemetry on every predict
# and does DNS lookups at startup. OFFLINE disables telemetry, the online
# checks and the auto-install; AUTOINSTALL is a second, independent guard.
# setdefault, so an operator can still override either deliberately.
# (HEIC support that auto-install was providing by accident is now a pinned
# dependency, registered explicitly in services/api/app/pipeline.py.)
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from .yolo_detection_service import YoloDetectionService  # noqa: E402

__all__ = ["YoloDetectionService"]
