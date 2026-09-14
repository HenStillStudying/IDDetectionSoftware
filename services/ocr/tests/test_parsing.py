from ktp_ocr.parsing import find_berlaku_hingga, split_tempat_tanggal_lahir
from ktp_ocr.text_lines import TextLine


def test_split_tempat_tanggal_lahir_with_comma():
    assert split_tempat_tanggal_lahir("KUPANG, 09-01-1989") == ("KUPANG", "09-01-1989")


def test_split_tempat_tanggal_lahir_with_misread_period():
    # Smaller OCR models sometimes misread the comma as a period.
    assert split_tempat_tanggal_lahir("KUPANG. 09-01-1989") == ("KUPANG", "09-01-1989")


def test_split_tempat_tanggal_lahir_no_separator_found():
    assert split_tempat_tanggal_lahir("KUPANG") == ("KUPANG", None)


def _line(text: str) -> TextLine:
    return TextLine(text=text, confidence=0.9, x1=0, y1=0, x2=10, y2=10)


def test_find_berlaku_hingga_with_colon():
    match = find_berlaku_hingga([_line("Berlaku Hingga: SEUMUR HIDUP")])
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_without_colon():
    # Smaller OCR models sometimes drop the colon entirely.
    match = find_berlaku_hingga([_line("Benaku HIngga SEUMUR HIDUP")])
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_with_heavily_garbled_word():
    # "Hingga" misread as "Hnga" (a dropped letter) inside an otherwise
    # badly garbled line ("Berlaku" -> "Betau") — an exact substring check
    # for "hingga" would miss this line entirely.
    match = find_berlaku_hingga([_line("Betau Hnga: SEUMUR HIDUP")])
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_absent():
    assert find_berlaku_hingga([_line("Agama"), _line("ISLAM")]) is None
