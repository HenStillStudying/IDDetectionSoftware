"""Per-field cleanup: splitting rows that carry two schema fields,
normalizing punctuation, and snapping enum-like fields to their nearest
known value from ktp_schema's reference data.
"""

from __future__ import annotations

import difflib
import re

from ktp_schema import BLOODS, PROVINCES

from .field_labels import find_same_row_value, find_value_line
from .text_lines import TextLine


def strip_value_prefix(text: str) -> str:
    """Strips a leading ': ' (or similar) that OCR keeps from 'Label: value' lines."""
    return re.sub(r"^[:\s]+", "", text).strip()


_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "I": "1", "l": "1", "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6"}
)


def normalize_digit_lookalikes(text: str) -> str:
    """Swaps letters OCR commonly misreads in place of a digit (O/o->0,
    I/l/i->1, S/s->5, B->8, Z/z->2, G->6) back to the digit — confirmed
    against a real KTP, where the blood-type letter "O" came back as the
    digit "0" (handled separately in split_jenis_kelamin_gol_darah, the
    reverse direction: a digit standing in for a letter). This is the
    mirror case: a letter standing in for a digit.

    Only call this on a value already known by its field's format to be
    digit-only or digit-heavy (NIK, RT/RW, an isolated date) — applying it
    to free text would corrupt real words that happen to contain these
    letters (e.g. "SEUMUR HIDUP" becoming "5EUMUR H1DUP").
    """
    return text.translate(_DIGIT_LOOKALIKES)


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", normalize_digit_lookalikes(text))


_TTL_SEPARATOR = re.compile(r"[,.]\s*(?=\d{1,2}[-\s]?\d{1,2}[-\s]?\d{4})")


def split_tempat_tanggal_lahir(value: str) -> tuple[str | None, str | None]:
    """'KUPANG, 09-01-1989' -> ('KUPANG', '09-01-1989').

    Splits on a comma OR period immediately before the date, since smaller
    OCR models sometimes misread the comma as a period — anchoring on the
    date pattern (rather than just the first comma) means that misread
    still splits correctly instead of silently merging both fields. The
    date's own internal separators (hyphen or space) are each optional, not
    just hyphen-or-space-required — a real KTP test first showed OCR
    dropping one of the two hyphens entirely ("13-03 2007", still
    space-separated), then a second test showed both separators collapsing
    with nothing between month and year at all ("13-032007"). A
    hyphen-or-space-required pattern recognized the first as a date but not
    the second, merging it whole into tempat_lahir instead; optional
    separators recognize both, plus the plain-hyphenated original.
    """
    match = _TTL_SEPARATOR.search(value)
    if not match:
        return (value.strip() or None, None)
    place = value[: match.start()].strip()
    date = value[match.end() :].strip()
    return (place or None, date or None)


def split_jenis_kelamin_gol_darah(value: str) -> tuple[str | None, str | None]:
    """'LAKI-LAKI   Gol. Darah: A' -> ('LAKI-LAKI', 'A').

    Matches a trailing '0' (digit zero) as well as 'O' (letter) — a real
    KTP test showed OCR misreading the blood-type letter "O" as the digit
    "0", which an ['ABO']-only pattern wouldn't recognize as a blood type
    at all, silently dropping it. Blood type is never actually a digit, so
    normalizing '0' to 'O' before snapping to the known BLOODS values is
    safe.
    """
    gol_match = re.search(r"\bgol\b", value, re.IGNORECASE)
    gender_part = value[: gol_match.start()] if gol_match else value
    blood_part = value[gol_match.start():] if gol_match else ""

    gender = gender_part.strip(" .:") or None

    blood_match = re.search(r"([AB0O]{1,2})\s*$", blood_part.upper())
    blood = snap_to_enum(blood_match.group(1).replace("0", "O"), BLOODS) if blood_match else None
    return (gender, blood)


def split_rt_rw_kelurahan(value: str) -> tuple[str | None, str | None]:
    """'010/001   Kel/Desa: SAMARINDA' -> ('010/001', 'SAMARINDA').

    Matches "kel" as a prefix, not a whole word (\\bkel, not \\bkel\\b): OCR
    sometimes drops the "/" in "Kel/Desa" entirely (e.g. "KELDESA"), which
    leaves no word boundary after "kel" for a whole-word match to find,
    silently returning the whole unsplit string instead.
    """
    kel_match = re.search(r"\bkel", value, re.IGNORECASE)
    if not kel_match:
        return (value.strip(" .:") or None, None)

    rt_rw = value[: kel_match.start()].strip(" .:") or None
    remainder = value[kel_match.start():]
    after_colon = remainder.split(":", 1)
    kelurahan = after_colon[1].strip() if len(after_colon) == 2 else None
    return (rt_rw, kelurahan)


def _best_match(value: str, choices: list[str]) -> tuple[str, float]:
    normalized = value.strip().upper()
    best = max(choices, key=lambda c: difflib.SequenceMatcher(None, normalized, c).ratio())
    score = difflib.SequenceMatcher(None, normalized, best).ratio()
    return best, score


def snap_to_enum(value: str, choices: list[str], threshold: float = 0.5) -> str | None:
    """Returns the closest known value to a noisy OCR'd string, or the
    stripped original if nothing is close enough to trust the snap.
    """
    normalized = value.strip().upper()
    if not normalized:
        return None
    best, score = _best_match(normalized, choices)
    return best if score >= threshold else normalized


def find_best_line_for_enum(
    lines: list[TextLine], choices: list[str], threshold: float = 0.6
) -> tuple[str, float] | None:
    """Finds the OCR line that best matches one of `choices` — used for
    header fields like provinsi that aren't preceded by their own label.
    """
    best_line: TextLine | None = None
    best_score = 0.0
    best_value: str | None = None
    for line in lines:
        value, score = _best_match(line.text, choices)
        if score > best_score:
            best_score, best_line, best_value = score, line, value
    if best_line is None or best_score < threshold:
        return None
    return best_value, best_line.confidence


_CITY_PREFIX = re.compile(r"^(kota\s*/?\s*kabupaten|kabupaten|kota)\s*", re.IGNORECASE)


def find_kota_kabupaten(lines: list[TextLine]) -> tuple[str, float] | None:
    for line in lines:
        match = _CITY_PREFIX.match(line.text.strip())
        if match:
            value = line.text[match.end():].strip()
            if value:
                return value, line.confidence
    return None


_HINGGA_REFERENCE = "hingga"
_HINGGA_MATCH_THRESHOLD = 0.5
_HINGGA_MIN_WORD_LENGTH = 4  # see note below on why this floor matters


def find_berlaku_hingga(
    lines: list[TextLine], card_size: tuple[int, int]
) -> tuple[str, float] | None:
    """Finds 'hingga' via fuzzy per-word matching, not an exact substring
    check — OCR noise can mangle it badly enough to lose a letter (e.g.
    'Hingga' -> 'Hnga', or the whole line to 'Betau Hnga: SEUMUR HIDUP'),
    which an exact 'in line.text' check would silently miss entirely rather
    than just misplacing the value.

    Once the label word is found, the value can be on the same line
    ("Berlaku Hingga: SEUMUR HIDUP", our synthetic layout), a separate box
    on the same row ("Berlaku Hingga" | "SEUMUR HIDUP" as two OCR boxes,
    confirmed to be how a real KTP prints it), or stacked below — the same
    three-way fallback every other field uses, built around field_labels'
    matchers since they only need a TextLine's bounds, not how it was found.

    Picks the single BEST-scoring word across every line, not the first one
    that merely clears the threshold: OCR line order isn't top-to-bottom
    (same reason field_labels.find_value_line uses geometry, not list
    order), so a coincidental match on an earlier, unrelated line — e.g.
    "TENGAH" (as in "JAWA TENGAH") scores 0.5 against "hingga" — could
    otherwise get returned before the real "Hingga" match later in the list
    is even considered.

    Short words are excluded before scoring: difflib's ratio is 2*matches /
    (len(a)+len(b)), so on a short word even a coincidental couple of shared
    letters clears a 0.5 threshold against a 6-letter reference — "GG" (as
    in the street prefix "GG. MAWAR") also matches at exactly 0.5. Real OCR
    variants of "Hingga" seen in practice are still >= 4 characters even
    when mangled.
    """
    best_score = 0.0
    best_line: TextLine | None = None
    best_word_index = -1

    for line in lines:
        words = line.text.split()
        for i, word in enumerate(words):
            normalized_word = re.sub(r"[^a-z]", "", word.lower())
            if len(normalized_word) < _HINGGA_MIN_WORD_LENGTH:
                continue
            score = difflib.SequenceMatcher(None, normalized_word, _HINGGA_REFERENCE).ratio()
            if score > best_score:
                best_score, best_line, best_word_index = score, line, i

    if best_line is None or best_score < _HINGGA_MATCH_THRESHOLD:
        return None

    inline_value = " ".join(best_line.text.split()[best_word_index + 1 :]).lstrip(" :.").strip()
    if inline_value:
        return inline_value, best_line.confidence

    same_row = find_same_row_value(lines, best_line)
    if same_row is not None:
        return same_row

    value_line = find_value_line(lines, best_line, card_size)
    if value_line is None:
        return None
    return strip_value_prefix(value_line.text), value_line.confidence


def find_provinsi(lines: list[TextLine]) -> tuple[str, float] | None:
    return find_best_line_for_enum(lines, PROVINCES)
