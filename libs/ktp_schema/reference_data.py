"""Canonical value lists for KTP enum-like fields.

Single source of truth shared by the synthetic dataset generator (which
picks from these to render fake cards) and the OCR parser (which snaps a
noisy recognized string to the nearest known value here). Keeping one list
instead of two means they can't silently drift apart.

NOTE: PROVINCES is a representative sample, not the complete list of all 38
Indonesian provinces — extend it if broader coverage is needed.
"""

PROVINCES = [
    "JAWA BARAT", "JAWA TENGAH", "JAWA TIMUR", "DKI JAKARTA",
    "BANTEN", "SUMATERA UTARA", "SULAWESI SELATAN", "KALIMANTAN TIMUR",
    "SUMATERA SELATAN", "RIAU", "ACEH", "BALI",
]

RELIGIONS = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDHA"]

JOBS = [
    "KARYAWAN SWASTA", "WIRASWASTA", "PELAJAR/MAHASISWA",
    "PETANI", "PNS", "BELUM/TIDAK BEKERJA", "DOKTER", "GURU",
]

BLOODS = ["A", "B", "AB", "O"]

MARITAL_STATUSES = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"]
