from ktp_ocr.field_labels import find_label_line, find_value_line
from ktp_ocr.text_lines import TextLine


def _line(text: str, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> TextLine:
    return TextLine(text=text, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)


def test_find_value_line_prefers_same_row_over_next_field_label():
    # A short label box and its (taller) value box can start at the exact
    # same y1 in OCR output. A strictly-positive vertical-gap requirement
    # would skip that real value and match the next field's label instead.
    label = _line("Alamat", x1=124, y1=177, x2=156, y2=188)
    value = _line(":JL. MERDEKA NO.47 RT 015/RW 001", x1=123, y1=177, x2=293, y2=202)
    next_field_label = _line("RTIRW", x1=124, y1=202, x2=159, y2=216)

    result = find_value_line([label, value, next_field_label], label, card_size=(640, 400))
    assert result is value


def test_find_value_line_still_finds_value_strictly_below_label():
    label = _line("Nama", x1=121, y1=95, x2=151, y2=110)
    value = _line(":REZA ANGGRAINI", x1=121, y1=100, x2=212, y2=121)

    result = find_value_line([label, value], label, card_size=(640, 400))
    assert result is value


def test_find_label_line_tolerates_ocr_noise():
    lines = [_line("Kecanatan", x1=0, y1=0, x2=50, y2=15)]  # "Kecamatan" misread
    match = find_label_line(lines, ["Kecamatan"])
    assert match is lines[0]
