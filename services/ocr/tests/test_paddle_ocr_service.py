import pytest

from ktp_ocr.paddle_ocr_service import PaddleOcrService


def test_onnxruntime_engine_requires_model_dirs():
    with pytest.raises(ValueError, match="requires text_detection_model_dir"):
        PaddleOcrService(engine="onnxruntime")


def test_onnxruntime_engine_requires_both_model_dirs_not_just_one():
    with pytest.raises(ValueError, match="requires text_detection_model_dir"):
        PaddleOcrService(engine="onnxruntime", text_detection_model_dir="/some/dir")


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="Unknown engine"):
        PaddleOcrService(engine="not-a-real-engine")
