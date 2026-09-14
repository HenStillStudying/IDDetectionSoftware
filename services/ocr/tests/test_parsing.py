from ktp_ocr.parsing import find_berlaku_hingga, split_rt_rw_kelurahan, split_tempat_tanggal_lahir
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


def test_find_berlaku_hingga_does_not_false_match_short_words():
    # "GG" (as in the street prefix "GG. MAWAR") scored exactly 0.5 against
    # "hingga" under difflib's ratio before a minimum-word-length guard was
    # added — grabbing an unrelated alamat line as the berlaku_hingga value.
    assert find_berlaku_hingga([_line("GG. MAWAR NO.26 RT 008/RW 003")]) is None


def test_find_berlaku_hingga_prefers_best_match_over_first_sufficient_one():
    # "TENGAH" (as in "JAWA TENGAH") also scores 0.5 against "hingga" and
    # can appear earlier in OCR's (not top-to-bottom) line order than the
    # real "Hingga" match — picking the first sufficient match instead of
    # the best one returned the province line's tail as the value.
    lines = [
        _line("JAWA TENGAH"),
        _line("Berlaku Hingga: SEUMUR HIDUP"),
    ]
    assert find_berlaku_hingga(lines) == ("SEUMUR HIDUP", 0.9)


def test_split_rt_rw_kelurahan_with_separator():
    assert split_rt_rw_kelurahan("010/001   Kel/Desa: SAMARINDA") == ("010/001", "SAMARINDA")


def test_split_rt_rw_kelurahan_when_ocr_drops_the_slash():
    # OCR sometimes reads "Kel/Desa" as "KELDESA" with no separator at all,
    # leaving no word boundary after "kel" for a \bkel\b match to find.
    assert split_rt_rw_kelurahan("007/005 KELDESA: CIMAHI") == ("007/005", "CIMAHI")
