"""The structured field contract shared by the OCR service and the API layer.

Every field extracted off a KTP is wrapped in `FieldValue` so a caller always
gets both the value and how much to trust it — a bare string would hide
whether "JAKARTA" was read cleanly or guessed from a blurry crop.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "LAKI-LAKI"
    FEMALE = "PEREMPUAN"


class MaritalStatus(str, Enum):
    SINGLE = "BELUM KAWIN"
    MARRIED = "KAWIN"
    DIVORCED = "CERAI HIDUP"
    WIDOWED = "CERAI MATI"


class FieldValue(BaseModel):
    """A single OCR'd field plus its recognition confidence."""

    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class KtpFields(BaseModel):
    """Every field present on a standard Indonesian KTP card."""

    nik: FieldValue
    nama: FieldValue
    tempat_lahir: FieldValue
    tanggal_lahir: FieldValue
    jenis_kelamin: FieldValue
    golongan_darah: FieldValue
    alamat: FieldValue
    rt_rw: FieldValue
    kelurahan_desa: FieldValue
    kecamatan: FieldValue
    agama: FieldValue
    status_perkawinan: FieldValue
    pekerjaan: FieldValue
    kewarganegaraan: FieldValue
    berlaku_hingga: FieldValue
    provinsi: FieldValue
    kota_kabupaten: FieldValue
