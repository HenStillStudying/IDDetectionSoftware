import datetime

import pytest

from ktp_schema.nik import NikGender, parse_nik, validate_nik

VALID_MALE_NIK = "3205121507900007"    # province 32, city 05, district 12, 15-07-1990, seq 0007
VALID_FEMALE_NIK = "3205125507900007"  # same but day+40 => female


def test_validate_nik_accepts_well_formed_nik():
    result = validate_nik(VALID_MALE_NIK)
    assert result.is_valid
    assert result.errors == []


def test_validate_nik_rejects_wrong_length():
    result = validate_nik("12345")
    assert not result.is_valid


def test_validate_nik_rejects_non_digit_characters():
    result = validate_nik("32051215079000A7")
    assert not result.is_valid


def test_validate_nik_rejects_zero_province_code():
    result = validate_nik("0005121507900007")
    assert not result.is_valid
    assert any("province" in e for e in result.errors)


def test_validate_nik_rejects_zero_sequence():
    result = validate_nik("3205121507900000")
    assert not result.is_valid
    assert any("sequence" in e for e in result.errors)


def test_validate_nik_rejects_invalid_month():
    result = validate_nik("3205121513900007")  # month=13
    assert not result.is_valid


def test_parse_nik_extracts_male_birth_date():
    info = parse_nik(VALID_MALE_NIK, reference_year=2024)
    assert info.gender == NikGender.MALE
    assert info.birth_date == datetime.date(1990, 7, 15)
    assert info.province_code == "32"
    assert info.city_code == "05"
    assert info.district_code == "12"
    assert info.sequence == "0007"


def test_parse_nik_extracts_female_birth_date_via_day_offset():
    info = parse_nik(VALID_FEMALE_NIK, reference_year=2024)
    assert info.gender == NikGender.FEMALE
    assert info.birth_date == datetime.date(1990, 7, 15)


def test_parse_nik_raises_on_invalid_nik():
    with pytest.raises(ValueError):
        parse_nik("not-a-nik")


def test_parse_nik_resolves_two_digit_year_century():
    # year=05 with reference_year=2024 (cutoff 24) => 2005, not 1905
    nik = "3205121507050007"
    info = parse_nik(nik, reference_year=2024)
    assert info.birth_date.year == 2005
