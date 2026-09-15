from ktp_ocr.field_labels import find_label_line, find_value_line, same_line_value
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


def test_find_label_line_matches_same_line_label_value_format():
    # A real KTP prints "Label : Value" as one OCR line — matching the
    # whole line against a short label like "NIK" used to score far below
    # threshold (diluted by the trailing value text). Only the portion
    # before the first colon should be compared.
    lines = [_line("NIK : 9205031303070001", x1=0, y1=0, x2=200, y2=15)]
    match = find_label_line(lines, ["NIK"])
    assert match is lines[0]


def test_same_line_value_extracts_value_after_colon():
    label_line = _line("NIK : 9205031303070001", x1=0, y1=0, x2=200, y2=15)
    assert same_line_value(label_line) == "9205031303070001"


def test_same_line_value_returns_none_for_stacked_layout_label():
    # The synthetic generator's label-only line has no colon at all.
    label_line = _line("Nama", x1=0, y1=0, x2=50, y2=15)
    assert same_line_value(label_line) is None


def test_same_line_value_returns_none_when_nothing_follows_colon():
    label_line = _line("Nama:", x1=0, y1=0, x2=50, y2=15)
    assert same_line_value(label_line) is None
