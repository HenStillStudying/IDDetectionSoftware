from .fields import Gender, MaritalStatus, KtpFields, FieldValue
from .nik import check_nik_consistency, validate_nik, NikInfo, parse_nik
from .result import BoundingBox, ForensicsResult, KtpExtractionResult, ExtractionStatus, NikConsistency
from .reference_data import BLOODS, JOBS, MARITAL_STATUSES, PROVINCES, RELIGIONS

__all__ = [
    "Gender",
    "MaritalStatus",
    "KtpFields",
    "FieldValue",
    "validate_nik",
    "check_nik_consistency",
    "parse_nik",
    "NikInfo",
    "BoundingBox",
    "ForensicsResult",
    "NikConsistency",
    "KtpExtractionResult",
    "ExtractionStatus",
    "PROVINCES",
    "RELIGIONS",
    "JOBS",
    "BLOODS",
    "MARITAL_STATUSES",
]
