"""NIK (Nomor Induk Kependudukan) structural parsing and validation.

NIK is 16 digits encoding administrative region and birth date:
    PP CC DDD DDMMYY SSSS
    -- -- --- ------ ----
    |  |  |   |       └─ 4-digit sequence, unique within province+city+district+dob
    |  |  |   └─ day(+40 if female)/month/2-digit year of birth
    |  |  └─ kecamatan (district) code
    |  └─ kota/kabupaten (city/regency) code
    └─ province code

This validates *structure*, not that the NIK belongs to a real registered
person — that requires a lookup against Dukcapil (the civil registry), which
is out of scope for this library.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from enum import Enum


# re.ASCII: \d is Unicode-aware by default, matching any Unicode decimal
# digit codepoint (e.g. U+FF10 FULLWIDTH DIGIT ZERO), not just 0-9. Without
# it, a NIK built from Unicode digit look-alikes could pass this pattern
# and validate_nik's exact-string "00" checks below while still evaluating
# to a real zero under int() — restricting to ASCII digits closes that gap.
NIK_PATTERN = re.compile(r"^\d{16}$", re.ASCII)


class NikGender(str, Enum):
    MALE = "LAKI-LAKI"
    FEMALE = "PEREMPUAN"


@dataclass(frozen=True)
class NikInfo:
    province_code: str
    city_code: str
    district_code: str
    birth_date: datetime.date
    gender: NikGender
    sequence: str


@dataclass(frozen=True)
class NikValidationResult:
    is_valid: bool
    errors: list[str]


def _resolve_birth_year(two_digit_year: int, reference_year: int | None = None) -> int:
    """NIK stores a 2-digit year with no century marker.

    Assumes the person is not older than ~100 years old: years greater than
    (reference_year - 2000) roll back into the 1900s, otherwise 2000s.
    """
    reference_year = reference_year or datetime.date.today().year
    century_cutoff = reference_year % 100
    return 2000 + two_digit_year if two_digit_year <= century_cutoff else 1900 + two_digit_year


def parse_nik(nik: str, reference_year: int | None = None) -> NikInfo:
    """Parses a 16-digit NIK into its structural components.

    Raises ValueError if the NIK is not structurally valid.
    """
    result = validate_nik(nik)
    if not result.is_valid:
        raise ValueError(f"Invalid NIK '{nik}': {'; '.join(result.errors)}")

    province_code = nik[0:2]
    city_code = nik[2:4]
    district_code = nik[4:6]
    day_raw = int(nik[6:8])
    month = int(nik[8:10])
    year_2digit = int(nik[10:12])
    sequence = nik[12:16]

    gender = NikGender.FEMALE if day_raw > 40 else NikGender.MALE
    day = day_raw - 40 if day_raw > 40 else day_raw
    year = _resolve_birth_year(year_2digit, reference_year)

    # validate_nik()'s day/month check above is deliberately lenient about
    # Feb 29 (it can't know the real century-resolved leap-year-ness from a
    # 2-digit year alone), so a NIK can pass it and still resolve here to a
    # year where Feb 29 doesn't exist — construct the date defensively
    # rather than let a bare ValueError escape as something that looks like
    # an unvalidated NIK slipped through.
    try:
        birth_date = datetime.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"Invalid NIK '{nik}': not a real calendar date ({exc})") from exc

    return NikInfo(
        province_code=province_code,
        city_code=city_code,
        district_code=district_code,
        birth_date=birth_date,
        gender=gender,
        sequence=sequence,
    )


def validate_nik(nik: str) -> NikValidationResult:
    """Validates the structural shape of a NIK without raising.

    Checks: length/digit format, non-zero region codes, a plausible
    day/month combination (accounting for the female +40 day offset), and a
    non-zero sequence number.
    """
    errors: list[str] = []

    if not isinstance(nik, str) or not NIK_PATTERN.match(nik):
        return NikValidationResult(is_valid=False, errors=["NIK must be exactly 16 digits"])

    province_code = nik[0:2]
    city_code = nik[2:4]
    district_code = nik[4:6]
    day_raw = int(nik[6:8])
    month = int(nik[8:10])
    sequence = nik[12:16]

    if province_code == "00":
        errors.append("province code cannot be 00")
    if city_code == "00":
        errors.append("city/regency code cannot be 00")
    if district_code == "00":
        errors.append("district code cannot be 00")

    day = day_raw - 40 if day_raw > 40 else day_raw
    if not (1 <= day <= 31):
        errors.append(f"day component out of range: {day}")
    if not (1 <= month <= 12):
        errors.append(f"month component out of range: {month}")
    if sequence == "0000":
        errors.append("sequence number cannot be 0000")

    # Reject impossible calendar dates (e.g. 31-02) using a lenient day/month
    # check only, since the century of the 2-digit year is ambiguous here.
    days_in_month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                      7: 31, 8: 30, 9: 30, 10: 31, 11: 30, 12: 31}
    if 1 <= month <= 12 and 1 <= day <= 31 and day > days_in_month[month]:
        errors.append(f"day {day} is invalid for month {month}")

    return NikValidationResult(is_valid=not errors, errors=errors)


def check_nik_consistency(
    nik: str, tanggal_lahir: str | None, jenis_kelamin: str | None
) -> tuple[list[str], list[str]]:
    """Cross-checks a structurally valid NIK against the birth date and
    gender printed elsewhere on the same card.

    Both are fixed for life and encoded in every genuine NIK (day of birth,
    +40 for women; month; 2-digit year), so a real card never disagrees with
    itself here — but a fabricated one easily can (every card produced by
    this project's own synthetic generator does, since it picks each field
    independently). Only birth date and gender are compared: the NIK's
    *region* codes are assigned once for life and don't change when someone
    moves or a province is split, so a genuine card can legitimately differ
    from its printed province.

    Returns (fields_checked, mismatches). A field that's missing or can't be
    parsed is skipped, never counted as a mismatch — an OCR miss must not
    look like a forgery. Only the year's last 2 digits are compared, since
    the NIK doesn't encode the century.
    """
    checked: list[str] = []
    mismatches: list[str] = []

    nik_day_raw = int(nik[6:8])
    nik_is_female = nik_day_raw > 40
    nik_day = nik_day_raw - 40 if nik_is_female else nik_day_raw
    nik_month = int(nik[8:10])
    nik_yy = nik[10:12]

    # By digits rather than an exact "DD-MM-YYYY" format: real-card OCR has
    # dropped one or both of the date's hyphens ("13-03 2007", "13-032007"),
    # and a genuine card must not be flagged for how OCR spaced its date.
    date_digits = re.sub(r"\D", "", tanggal_lahir or "", flags=re.ASCII)
    if len(date_digits) == 8:
        checked.append("tanggal_lahir")
        day, month, yy = int(date_digits[0:2]), int(date_digits[2:4]), date_digits[6:8]
        if (day, month, yy) != (nik_day, nik_month, nik_yy):
            mismatches.append(
                f"printed birth date {date_digits[0:2]}-{date_digits[2:4]}-{date_digits[4:8]} "
                f"does not match the NIK's encoded birth date "
                f"{nik_day:02d}-{nik_month:02d}-**{nik_yy}"
            )

    gender = (jenis_kelamin or "").strip().upper()
    if gender in (NikGender.MALE.value, NikGender.FEMALE.value):
        checked.append("jenis_kelamin")
        printed_is_female = gender == NikGender.FEMALE.value
        if printed_is_female != nik_is_female:
            nik_gender = NikGender.FEMALE if nik_is_female else NikGender.MALE
            mismatches.append(
                f"printed gender {gender} does not match the NIK's encoded gender {nik_gender.value}"
            )

    return checked, mismatches
