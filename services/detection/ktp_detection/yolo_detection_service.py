"""Real DetectionService implementation backed by a trained YOLO model.

NOTE on rectification: the model predicts an axis-aligned bounding box, not
the card's four corners, so localization alone leaves a tilted card tilted
in the crop. `_deskew` closes that gap with classic CV rather than a learned
model: it finds the card's contour within the padded crop, reduces it to its
four corner points, and applies a perspective warp to flatten it to an
upright rectangle. This handles true perspective distortion (a card
photographed at an angle, not just rotated in-plane) as well as simple
rotation, since a rotated rectangle is just a special case of a quadrilateral
with four corners.

If a clean 4-point quadrilateral can't be resolved (rounded corners,
occlusion, background clutter), it falls back to a rotation-only correction
(minAreaRect-based) rather than leaving the card uncorrected. This is still
classic CV, not a learned model, so it will be less robust than a trained
keypoint model on cluttered real-world backgrounds — that's the next step up
if this proves insufficient.
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
QUAD_APPROX_EPSILON_FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.08)  # tried in order until 4 points found


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
    def _find_card_quadrilateral(contour: np.ndarray) -> np.ndarray | None:
        """Reduces a contour to its 4 corner points via polygon
        approximation, trying progressively coarser tolerances until exactly
        4 points remain. Returns None if no tolerance in the search yields a
        clean quadrilateral (rounded corners, noisy edges, occlusion).
        """
        perimeter = cv2.arcLength(contour, True)
        for epsilon_fraction in QUAD_APPROX_EPSILON_FRACTIONS:
            approx = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        return None

    @staticmethod
    def _order_corners(points: np.ndarray) -> np.ndarray:
        """Orders 4 points as [top-left, top-right, bottom-right, bottom-left].

        Relies on the card not being rotated anywhere near 45 degrees within
        the crop (true for our detector's axis-aligned box plus a modest
        margin) — at extreme rotations this sum/difference heuristic can
        mislabel which corner is "top-left".
        """
        ordered = np.zeros((4, 2), dtype=np.float32)

        total = points.sum(axis=1)
        ordered[0] = points[np.argmin(total)]  # top-left: smallest x+y
        ordered[2] = points[np.argmax(total)]  # bottom-right: largest x+y

        diff = np.diff(points, axis=1).flatten()  # y - x
        ordered[1] = points[np.argmin(diff)]  # top-right: smallest y-x
        ordered[3] = points[np.argmax(diff)]  # bottom-left: largest y-x

        return ordered

    @staticmethod
    def _expand_corners_outward(corners: np.ndarray, margin_ratio: float = 0.02) -> np.ndarray:
        """Pushes each corner slightly outward from the quad's centroid.

        cv2.approxPolyDP (and the Canny+dilate edge detection feeding it)
        tends to round off sharp corners slightly inward, which was cropping
        a few real pixels off the card's edge — enough to clip the top
        header row of text entirely on some cards. A small outward margin
        trades a thin sliver of extra background for not losing real
        content; the padded crop this operates on already has slack around
        the card (CROP_MARGIN_RATIO) for this to land within.
        """
        centroid = corners.mean(axis=0)
        expanded = centroid + (corners - centroid) * (1 + margin_ratio)
        return expanded.astype(np.float32)

    @staticmethod
    def _warp_to_flat_rectangle(image: PILImage, corners: np.ndarray) -> PILImage:
        top_left, top_right, bottom_right, bottom_left = corners

        width_top = np.hypot(*(top_right - top_left))
        width_bottom = np.hypot(*(bottom_right - bottom_left))
        target_width = max(int(width_top), int(width_bottom))

        height_left = np.hypot(*(bottom_left - top_left))
        height_right = np.hypot(*(bottom_right - top_right))
        target_height = max(int(height_left), int(height_right))

        destination = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(corners, destination)
        warped = cv2.warpPerspective(np.array(image), matrix, (target_width, target_height))
        return Image.fromarray(warped)

    def _deskew(self, padded_crop: PILImage) -> PILImage:
        contour = self._find_largest_card_contour(padded_crop)
        if contour is None:
            return padded_crop  # no confident card-shaped edge found; leave as-is

        quad = self._find_card_quadrilateral(contour)
        if quad is not None:
            ordered_corners = self._order_corners(quad)
            ordered_corners = self._expand_corners_outward(ordered_corners)
            return self._warp_to_flat_rectangle(padded_crop, ordered_corners)

        return self._rotation_only_deskew(padded_crop, contour)

    def _rotation_only_deskew(self, padded_crop: PILImage, contour: np.ndarray) -> PILImage:
        """Fallback for when a clean 4-point quadrilateral couldn't be
        resolved — corrects in-plane rotation only, not perspective.
        """
        angle = self._rotation_angle_degrees(contour)
        rotated = padded_crop.rotate(angle, expand=True, fillcolor=(0, 0, 0), resample=Image.BICUBIC)

        tight_contour = self._find_largest_card_contour(rotated)
        if tight_contour is None:
            return rotated
        x, y, w, h = cv2.boundingRect(tight_contour)
        return rotated.crop((x, y, x + w, y + h))

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
