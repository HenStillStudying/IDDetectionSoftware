"""Real DetectionService implementation backed by a trained YOLO model.

NOTE on rectification: the model predicts an axis-aligned bounding box, not
the card's four corners, so localization alone leaves a rotated card rotated
in the crop. `_deskew` closes most of that gap with classic CV (contour +
minAreaRect) rather than a learned model: it finds the card's edge within the
padded crop, measures its tilt, and rotates it upright. This still isn't true
perspective correction (a card photographed at a steep angle will be
stretched, not just rotated) — that needs a 4-corner keypoint model — but it
handles the common case of a flat card photographed at a rotation, which is
what our detector's axis-aligned box otherwise leaves uncorrected.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from ultralytics import YOLO

from ktp_interfaces import DetectionService
from ktp_schema import BoundingBox

DEFAULT_CONFIDENCE_THRESHOLD = 0.4
CROP_MARGIN_RATIO = 0.05  # extra margin around the predicted box, as a fraction of its size
MIN_CONTOUR_AREA_RATIO = 0.15  # skip deskewing if no confident card-shaped contour is found


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
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, detection_confidence=confidence)

        margin_x1, margin_y1, margin_x2, margin_y2 = self._add_margin(x1, y1, x2, y2, image.size)
        padded_crop = image.crop((margin_x1, margin_y1, margin_x2, margin_y2))

        upright = self._deskew(padded_crop)
        return upright, bbox

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

    @staticmethod
    def _find_largest_card_contour(image: PILImage) -> np.ndarray | None:
        gray = np.array(image.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (gray.shape[0] * gray.shape[1])
        if area_ratio < MIN_CONTOUR_AREA_RATIO:
            return None
        return largest

    @staticmethod
    def _rotation_angle_degrees(contour: np.ndarray) -> float:
        """Minimal rotation (degrees) needed to make the contour's longest
        edge horizontal, normalized to (-45, 45] so we always rotate the
        short way rather than flipping the card upside down or sideways.
        """
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        p1, p2 = max(edges, key=lambda e: np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))

        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
        if angle > 90:
            angle -= 180
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        return angle

    def _deskew(self, padded_crop: PILImage) -> PILImage:
        contour = self._find_largest_card_contour(padded_crop)
        if contour is None:
            return padded_crop  # no confident card-shaped edge found; leave as-is

        angle = self._rotation_angle_degrees(contour)
        rotated = padded_crop.rotate(angle, expand=True, fillcolor=(0, 0, 0), resample=Image.BICUBIC)

        tight_contour = self._find_largest_card_contour(rotated)
        if tight_contour is None:
            return rotated
        x, y, w, h = cv2.boundingRect(tight_contour)
        return rotated.crop((x, y, x + w, y + h))
