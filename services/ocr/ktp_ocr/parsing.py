"""Per-field cleanup: splitting rows that carry two schema fields,
normalizing punctuation, and snapping enum-like fields to their nearest
known value from ktp_schema's reference data.
"""

from __future__ import annotations

import difflib
import re

from ktp_schema import BLOODS, PROVINCES

from .text_lines import TextLine


def strip_value_prefix(text: str) -> str:
    """Strips a leading ': ' (or similar) that OCR keeps from 'Label: value' lines."""
    return re.sub(r"^[:\s]+", "", text).strip()


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


_TTL_SEPARATOR = re.compile(r"[,.]\s*(?=\d{1,2}-\d{1,2}-\d{4})")


def split_tempat_tanggal_lahir(value: str) -> tuple[str | None, str | None]:
    """'KUPANG, 09-01-1989' -> ('KUPANG', '09-01-1989').

    Splits on a comma OR period immediately before the date, since smaller
    OCR models sometimes misread the comma as a period — anchoring on the
    date pattern (rather than just the first comma) means that misread
    still splits correctly instead of silently merging both fields.
    """
    match = _TTL_SEPARATOR.search(value)
    if not match:
        return (value.strip() or None, None)
    place = value[: match.start()].strip()
    date = value[match.end() :].strip()
    return (place or None, date or None)


def split_jenis_kelamin_gol_darah(value: str) -> tuple[str | None, str | None]:
    """'LAKI-LAKI   Gol. Darah: A' -> ('LAKI-LAKI', 'A')."""
    gol_match = re.search(r"\bgol\b", value, re.IGNORECASE)
    gender_part = value[: gol_match.start()] if gol_match else value
    blood_part = value[gol_match.start():] if gol_match else ""

    gender = gender_part.strip(" .:") or None

    blood_match = re.search(r"([ABO]{1,2})\s*$", blood_part.upper())
    blood = snap_to_enum(blood_match.group(1), BLOODS) if blood_match else None
    return (gender, blood)


def split_rt_rw_kelurahan(value: str) -> tuple[str | None, str | None]:
    """'010/001   Kel/Desa: SAMARINDA' -> ('010/001', 'SAMARINDA')."""
    kel_match = re.search(r"\bkel\b", value, re.IGNORECASE)
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


_BERLAKU_PREFIX = re.compile(r"^.*?hingga\s*[:.]?\s*", re.IGNORECASE)


def find_berlaku_hingga(lines: list[TextLine]) -> tuple[str, float] | None:
    """Strips everything through 'hingga' (plus an optional colon), rather
    than requiring a literal colon — smaller OCR models sometimes drop it
    (e.g. 'Benaku HIngga SEUMUR HIDUP' with no ':'), which previously left
    the label text leaking into the value.
    """
    for line in lines:
        if "hingga" in line.text.lower():
            value = _BERLAKU_PREFIX.sub("", line.text, count=1).strip()
            if value:
                return value, line.confidence
    return None


def find_provinsi(lines: list[TextLine]) -> tuple[str, float] | None:
    return find_best_line_for_enum(lines, PROVINCES)
