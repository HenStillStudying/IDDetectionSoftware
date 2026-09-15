"""Canonical value lists for KTP enum-like fields.

Single source of truth shared by the synthetic dataset generator (which
picks from these to render fake cards) and the OCR parser (which snaps a
noisy recognized string to the nearest known value here). Keeping one list
instead of two means they can't silently drift apart.

PROVINCES was previously a 12-item sample, not the full official list —
found incomplete when a real KTP (Papua Barat Daya, one of the provinces
created in 2022) failed to match anything. Now the complete list of 38.
RELIGIONS was similarly missing Konghucu, one of the six officially
recognized religions on an Indonesian KTP.
"""

PROVINCES = [
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "KEPULAUAN RIAU",
    "JAMBI", "SUMATERA SELATAN", "BANGKA BELITUNG", "BENGKULU", "LAMPUNG",
    "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR",
    "BANTEN", "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR",
    "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
    "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA",
    "SULAWESI TENGAH", "SULAWESI SELATAN", "SULAWESI TENGGARA", "GORONTALO",
    "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA", "PAPUA BARAT",
    "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN", "PAPUA BARAT DAYA",
]

RELIGIONS = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDHA", "KONGHUCU"]

JOBS = [
    "KARYAWAN SWASTA", "WIRASWASTA", "PELAJAR/MAHASISWA",
    "PETANI", "PNS", "BELUM/TIDAK BEKERJA", "DOKTER", "GURU",
]

BLOODS = ["A", "B", "AB", "O"]

MARITAL_STATUSES = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"]
