from .fields import Gender, MaritalStatus, KtpFields, FieldValue
from .nik import validate_nik, NikInfo, parse_nik
from .result import BoundingBox, ForensicsResult, KtpExtractionResult, ExtractionStatus
from .reference_data import BLOODS, JOBS, MARITAL_STATUSES, PROVINCES, RELIGIONS

__all__ = [
    "Gender",
    "MaritalStatus",
    "KtpFields",
    "FieldValue",
    "validate_nik",
    "parse_nik",
    "NikInfo",
    "BoundingBox",
    "ForensicsResult",
    "KtpExtractionResult",
    "ExtractionStatus",
    "PROVINCES",
    "RELIGIONS",
    "JOBS",
    "BLOODS",
    "MARITAL_STATUSES",
]
