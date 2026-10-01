from ktp_ocr.parsing import (
    find_berlaku_hingga,
    normalize_digit_lookalikes,
    only_digits,
    parse_nik,
    split_jenis_kelamin_gol_darah,
    split_rt_rw_kelurahan,
    split_tempat_tanggal_lahir,
)
from ktp_ocr.field_labels import merged_label_prefix_value
from ktp_ocr.text_lines import TextLine

CARD_SIZE = (640, 400)


def test_split_jenis_kelamin_gol_darah_normal_case():
    assert split_jenis_kelamin_gol_darah("LAKI-LAKI   Gol. Darah: A") == ("LAKI-LAKI", "A")


def test_split_jenis_kelamin_gol_darah_with_digit_zero_misread_as_letter_o():
    # A real KTP test showed OCR reading the blood-type letter "O" as the
    # digit "0" instead.
    assert split_jenis_kelamin_gol_darah("LAKILAKI Gol. Derah 0") == ("LAKILAKI", "O")


def test_normalize_digit_lookalikes_swaps_letters_back_to_digits():
    assert normalize_digit_lookalikes("13-03-2OO7") == "13-03-2007"
    assert normalize_digit_lookalikes("00I/002") == "001/002"


def test_normalize_digit_lookalikes_leaves_non_digit_context_untouched_by_caller_choice():
    # The function itself has no way to know context -- it's the caller's
    # job to only invoke it on values already known to be digit-only/heavy
    # by field format. Documented here as a warning, not a guarantee: this
    # WOULD corrupt free text if misapplied.
    assert normalize_digit_lookalikes("SEUMUR HIDUP") == "5EUMUR H1DUP"


def test_only_digits_corrects_letter_lookalikes_instead_of_dropping_them():
    # Previously this just stripped any non-digit character, which would
    # silently shorten a NIK (and break its length validation) if OCR
    # misread even one digit as a letter, instead of correcting it.
    assert only_digits("92O5031303070001") == "9205031303070001"


def test_split_tempat_tanggal_lahir_with_comma():
    assert split_tempat_tanggal_lahir("KUPANG, 09-01-1989") == ("KUPANG", "09-01-1989")


def test_split_tempat_tanggal_lahir_with_misread_period():
    # Smaller OCR models sometimes misread the comma as a period.
    assert split_tempat_tanggal_lahir("KUPANG. 09-01-1989") == ("KUPANG", "09-01-1989")


def test_split_tempat_tanggal_lahir_no_separator_found():
    assert split_tempat_tanggal_lahir("KUPANG") == ("KUPANG", None)


def test_split_tempat_tanggal_lahir_with_dropped_hyphen_in_date():
    # OCR dropped the hyphen before the year on a real KTP.
    assert split_tempat_tanggal_lahir("SORONG, 13-03 2007") == ("SORONG", "13-03 2007")


def test_split_tempat_tanggal_lahir_with_both_separators_dropped():
    # A second real KTP test showed OCR collapsing both the space AND the
    # hyphen between month and year, leaving no separator there at all.
    assert split_tempat_tanggal_lahir("SORONG,13-032007") == ("SORONG", "13-032007")


def _line(text: str, x1: float = 0, y1: float = 0, x2: float = 10, y2: float = 10) -> TextLine:
    return TextLine(text=text, confidence=0.9, x1=x1, y1=y1, x2=x2, y2=y2)


def test_find_berlaku_hingga_with_colon():
    match = find_berlaku_hingga([_line("Berlaku Hingga: SEUMUR HIDUP")], CARD_SIZE)
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_without_colon():
    # Smaller OCR models sometimes drop the colon entirely.
    match = find_berlaku_hingga([_line("Benaku HIngga SEUMUR HIDUP")], CARD_SIZE)
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_with_heavily_garbled_word():
    # "Hingga" misread as "Hnga" (a dropped letter) inside an otherwise
    # badly garbled line ("Berlaku" -> "Betau") — an exact substring check
    # for "hingga" would miss this line entirely.
    match = find_berlaku_hingga([_line("Betau Hnga: SEUMUR HIDUP")], CARD_SIZE)
    assert match == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_absent():
    assert find_berlaku_hingga([_line("Agama"), _line("ISLAM")], CARD_SIZE) is None


def test_find_berlaku_hingga_does_not_false_match_short_words():
    # "GG" (as in the street prefix "GG. MAWAR") scored exactly 0.5 against
    # "hingga" under difflib's ratio before a minimum-word-length guard was
    # added — grabbing an unrelated alamat line as the berlaku_hingga value.
    assert find_berlaku_hingga([_line("GG. MAWAR NO.26 RT 008/RW 003")], CARD_SIZE) is None


def test_find_berlaku_hingga_prefers_best_match_over_first_sufficient_one():
    # "TENGAH" (as in "JAWA TENGAH") also scores 0.5 against "hingga" and
    # can appear earlier in OCR's (not top-to-bottom) line order than the
    # real "Hingga" match — picking the first sufficient match instead of
    # the best one returned the province line's tail as the value.
    lines = [
        _line("JAWA TENGAH"),
        _line("Berlaku Hingga: SEUMUR HIDUP"),
    ]
    assert find_berlaku_hingga(lines, CARD_SIZE) == ("SEUMUR HIDUP", 0.9)


def test_find_berlaku_hingga_from_separate_box_on_the_same_row():
    # A real KTP prints the label and value as two separate OCR boxes on
    # the same row, not merged into one line — confirmed against a real
    # card, where this was the dominant layout for every field.
    label = _line("Berlaku Hingga", x1=60, y1=800, x2=327, y2=850)
    value = _line("SEUMUR HIDUP", x1=397, y1=808, x2=703, y2=846)
    assert find_berlaku_hingga([label, value], CARD_SIZE) == ("SEUMUR HIDUP", 0.9)


def test_split_rt_rw_kelurahan_with_separator():
    assert split_rt_rw_kelurahan("010/001   Kel/Desa: SAMARINDA") == ("010/001", "SAMARINDA")


def test_split_rt_rw_kelurahan_when_ocr_drops_the_slash():
    # OCR sometimes reads "Kel/Desa" as "KELDESA" with no separator at all,
    # leaving no word boundary after "kel" for a \bkel\b match to find.
    assert split_rt_rw_kelurahan("007/005 KELDESA: CIMAHI") == ("007/005", "CIMAHI")


# --- NIK: a misread colon must not become a leading digit ---
_NIK = "9205031303070001"


def test_parse_nik_plain():
    assert parse_nik(_NIK) == _NIK


def test_parse_nik_drops_colon_misread_as_a_separate_leading_character():
    # A real blurry photo: the ":" after "NIK" came back as a look-alike
    # letter, which digit normalization then turned into a leading 1,
    # giving a 17-digit NIK.
    for misread in ("l", "I", "i", "|", "1", "!", ";"):
        assert parse_nik(f"{misread} {_NIK}") == _NIK, misread


def test_parse_nik_through_the_merged_label_path():
    # With no ":" left on the line, it's the merged label+value handler that
    # hands parse_nik the stray character.
    line = TextLine(text=f"NIK l {_NIK}", confidence=0.9, x1=0, y1=0, x2=300, y2=15)
    assert parse_nik(merged_label_prefix_value(line, ["NIK"])) == _NIK


def test_parse_nik_leaves_other_wrong_lengths_alone():
    # Only the specific "one stray separator in front of 16 digits" shape is
    # repaired; anything else stays as read, for validation to reject.
    assert parse_nik(f"1{_NIK}") == "1" + _NIK  # no gap: can't tell it's a separator
    assert parse_nik(f"l {_NIK}5") == "1" + _NIK + "5"  # 18 digits
    assert parse_nik(_NIK[:-1]) == _NIK[:-1]  # 15 digits
