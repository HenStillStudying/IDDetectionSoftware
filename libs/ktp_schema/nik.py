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


NIK_PATTERN = re.compile(r"^\d{16}$")


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

    birth_date = datetime.date(year, month, day)

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
