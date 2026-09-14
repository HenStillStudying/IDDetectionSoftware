"""Real DetectionService implementation backed by a trained YOLO model.

NOTE on rectification: the model predicts an axis-aligned bounding box, not
the card's four corners, so "rectify" here only means "crop to the box with
a small margin" — a rotated card still comes out rotated in the crop. True
perspective correction (warping to a flat upright rectangle) needs either an
oriented-bounding-box model or a 4-corner keypoint model; that's a follow-up,
not something this bbox-only detector can do.
"""

from __future__ import annotations

from PIL.Image import Image as PILImage
from ultralytics import YOLO

from ktp_interfaces import DetectionService
from ktp_schema import BoundingBox

DEFAULT_CONFIDENCE_THRESHOLD = 0.4
CROP_MARGIN_RATIO = 0.03  # extra margin around the predicted box, as a fraction of its size


class YoloDetectionService(DetectionService):
    def __init__(self, weights_path: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold

    def detect_and_rectify(self, image: PILImage) -> tuple[PILImage, BoundingBox] | None:
        results = self._model.predict(image, verbose=False)[0]
        if len(results.boxes) == 0:
            return None

        best = max(results.boxes, key=lambda box: float(box.conf[0]))
        confidence = float(best.conf[0])
        if confidence < self._confidence_threshold:
            return None

        x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
        x1, y1, x2, y2 = self._add_margin(x1, y1, x2, y2, image.size)

        cropped = image.crop((x1, y1, x2, y2))
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, detection_confidence=confidence)
        return cropped, bbox

    @staticmethod
    def _add_margin(
        x1: int, y1: int, x2: int, y2: int, image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        margin_x = int((x2 - x1) * CROP_MARGIN_RATIO)
        margin_y = int((y2 - y1) * CROP_MARGIN_RATIO)
        return (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )
