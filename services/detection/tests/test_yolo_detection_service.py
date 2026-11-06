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


KTP, OTHER = 0, 1


class _FakeBox:
    def __init__(self, xyxy, conf, cls=KTP):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([float(cls)])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes, names):
        self._boxes = boxes
        self.names = names

    def predict(self, image, verbose=False):
        return [_FakeResult(self._boxes)]


def _service(boxes, names=None) -> YoloDetectionService:
    """`boxes` are (xyxy, conf) or (xyxy, conf, cls). The default model is
    single-class, like the production detector."""
    service = YoloDetectionService.__new__(YoloDetectionService)
    service._model = _FakeModel([_FakeBox(*box) for box in boxes], names or {KTP: "ktp"})
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


# --- Two-class models: "ktp" plus "other_document" (any other ID card) ---
_TWO_CLASS = {KTP: "ktp", OTHER: "other_document"}


def test_other_document_alone_is_no_card():
    result = _service([(_LEFT_CARD, 0.95, OTHER)], _TWO_CLASS).detect_and_rectify(_IMAGE)
    assert result is None


def test_ktp_next_to_another_document_is_refused():
    # Photograph one card at a time: a SIM beside the KTP counts as a card.
    service = _service([(_LEFT_CARD, 0.95, KTP), (_RIGHT_CARD, 0.9, OTHER)], _TWO_CLASS)
    with pytest.raises(MultipleCardsDetectedError) as excinfo:
        service.detect_and_rectify(_IMAGE)
    assert excinfo.value.count == 2


def test_one_card_boxed_as_both_classes_takes_the_more_confident():
    ktp_wins = _service([(_LEFT_CARD, 0.9, KTP), (_LEFT_CARD, 0.5, OTHER)], _TWO_CLASS)
    assert ktp_wins.detect_and_rectify(_IMAGE) is not None
    other_wins = _service([(_LEFT_CARD, 0.9, OTHER), (_LEFT_CARD, 0.5, KTP)], _TWO_CLASS)
    assert other_wins.detect_and_rectify(_IMAGE) is None
