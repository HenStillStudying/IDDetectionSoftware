"""Tests the card-counting logic around the YOLO model, with the model
itself faked out, so no trained weights are needed. Skipped where
ultralytics isn't installed (CI installs only the light dependencies).
"""

import pytest

pytest.importorskip("ultralytics")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ktp_detection import YoloDetectionService  # noqa: E402
from ktp_interfaces import MultipleCardsDetectedError  # noqa: E402


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, image, verbose=False):
        return [_FakeResult(self._boxes)]


def _service(boxes) -> YoloDetectionService:
    service = YoloDetectionService.__new__(YoloDetectionService)
    service._model = _FakeModel([_FakeBox(xyxy, conf) for xyxy, conf in boxes])
    service._confidence_threshold = 0.4
    return service


_IMAGE = Image.new("RGB", (1600, 900), color=(120, 120, 120))
_LEFT_CARD = (40, 120, 740, 560)
_RIGHT_CARD = (860, 120, 1560, 560)


def test_single_card_is_returned():
    result = _service([(_LEFT_CARD, 0.96)]).detect_and_rectify(_IMAGE)
    assert result is not None
    assert result[1].x1 == 40


def test_two_separate_cards_are_refused():
    service = _service([(_LEFT_CARD, 0.96), (_RIGHT_CARD, 0.95)])
    with pytest.raises(MultipleCardsDetectedError) as excinfo:
        service.detect_and_rectify(_IMAGE)
    assert excinfo.value.count == 2


def test_second_box_below_threshold_is_ignored():
    result = _service([(_LEFT_CARD, 0.96), (_RIGHT_CARD, 0.2)]).detect_and_rectify(_IMAGE)
    assert result is not None


def test_overlapping_duplicate_box_on_one_card_is_not_a_second_card():
    # A second box mostly inside the first (e.g. around part of the same
    # card) is one card seen twice, not two cards.
    inner = (60, 140, 600, 500)
    result = _service([(_LEFT_CARD, 0.96), (inner, 0.5)]).detect_and_rectify(_IMAGE)
    assert result is not None
