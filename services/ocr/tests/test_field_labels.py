from ktp_ocr.field_labels import find_label_line, find_same_row_value, find_value_line, same_line_value
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


def test_find_same_row_value_excludes_next_row_value_bleeding_upward():
    # Coordinates from a real synthetic-card failure: an all-caps value's
    # OCR box reaches full cap-height while its mixed-case label's doesn't,
    # so the *next* row's value box can start high enough to also clear
    # "Jenis Kelamin"'s row-overlap check — this used to concatenate
    # "Alamat"'s value in too, silently breaking golongan_darah's
    # end-anchored blood-type parsing downstream.
    label = _line("Jenis Kelamin", x1=128, y1=123, x2=188, y2=140)
    own_value = _line(": PEREMPUAN Gol. Darah: AB", x1=228, y1=111, x2=377, y2=134)
    next_row_value = _line(
        ": JL. SUDIRMAN NO.149 RT 001/RW 005", x1=228, y1=125, x2=416, y2=152
    )

    result = find_same_row_value([label, own_value, next_row_value], label)
    assert result is not None
    value, _ = result
    assert "SUDIRMAN" not in value
    assert value == "PEREMPUAN Gol. Darah: AB"


def test_find_same_row_value_prefers_above_over_nearest_on_near_tie():
    # Coordinates from a real synthetic-card failure: NIK's own value sits
    # slightly *above* the NIK label's center, and Nama's value (the next
    # row down) happened to sit only marginally closer to the label's
    # center by raw distance. Picking by nearest-distance alone chose the
    # wrong one (Nama's value has no digits, so NIK came back empty); the
    # true value should win because it's the at-or-above candidate.
    label = _line("NIK", x1=126, y1=66, x2=150, y2=83)
    own_value = _line(":3455532505839582", x1=225, y1=49, x2=368, y2=78)
    next_row_value = _line(": JASMIN MARYADI, S.T.", x1=225, y1=72, x2=343, y2=98)

    result = find_same_row_value([label, own_value, next_row_value], label)
    assert result is not None
    value, _ = result
    assert value == "3455532505839582"
