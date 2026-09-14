"""Real OcrService implementation backed by PaddleOCR.

Reads the whole rectified card with a general OCR engine, then locates each
field by fuzzy-matching known Indonesian KTP labels and reading the value
positioned below each label (see field_labels.py for why geometry, not OCR
line order, is what locates the value). This tolerates the imperfect
rectification produced by YoloDetectionService — a still-somewhat-rotated
or shadow-marked card — far better than a fixed-pixel-region crop would.
"""

from __future__ import annotations

from PIL.Image import Image as PILImage

from ktp_interfaces import OcrService
from ktp_schema import FieldValue, Gender, JOBS, KtpFields, MARITAL_STATUSES, RELIGIONS

GENDERS = [Gender.MALE.value, Gender.FEMALE.value]

from .field_labels import ROW_LABELS, find_label_line, find_value_line
from .parsing import (
    find_berlaku_hingga,
    find_kota_kabupaten,
    find_provinsi,
    only_digits,
    snap_to_enum,
    split_jenis_kelamin_gol_darah,
    split_rt_rw_kelurahan,
    split_tempat_tanggal_lahir,
    strip_value_prefix,
)
from .text_lines import TextLine, run_ocr


def _row_value(
    lines: list[TextLine], row_key: str, card_size: tuple[int, int]
) -> tuple[str | None, float]:
    label_line = find_label_line(lines, ROW_LABELS[row_key])
    if label_line is None:
        return None, 0.0
    value_line = find_value_line(lines, label_line, card_size)
    if value_line is None:
        return None, 0.0
    return strip_value_prefix(value_line.text), value_line.confidence


def _field(value: str | None, confidence: float) -> FieldValue:
    return FieldValue(value=value, confidence=confidence if value else 0.0)


def _field_from_match(match: tuple[str, float] | None) -> FieldValue:
    if match is None:
        return FieldValue(value=None, confidence=0.0)
    value, confidence = match
    return FieldValue(value=value, confidence=confidence)


# PP-OCRv6_small_* over the default PP-OCRv6_medium_*: ~4-5x faster
# (~12s -> ~3s/request) with NIK recognition holding up correctly even on
# heavily degraded test images. PP-OCRv6_tiny_* was faster still but
# dropped NIK entirely on the hardest test case and started merging
# adjacent text lines into single detection boxes (undermining the
# position-based field matching this service relies on) — not an
# acceptable tradeoff for an ID-verification field.
DEFAULT_TEXT_DETECTION_MODEL = "PP-OCRv6_small_det"
DEFAULT_TEXT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"


class PaddleOcrService(OcrService):
    def __init__(
        self,
        lang: str = "en",
        text_detection_model_name: str = DEFAULT_TEXT_DETECTION_MODEL,
        text_recognition_model_name: str = DEFAULT_TEXT_RECOGNITION_MODEL,
    ):
        from paddleocr import PaddleOCR

        # Skip doc-orientation/unwarping/textline-orientation: YoloDetectionService
        # already deskews the card upstream, so these three extra model
        # stages would be redundant work on every request.
        #
        # enable_mkldnn=False works around a PaddlePaddle/oneDNN crash seen
        # on this stack (NotImplementedError in onednn_instruction.cc);
        # revisit if a paddlepaddle upgrade fixes the underlying bug.
        self._engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            enable_mkldnn=False,
        )

    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        lines = run_ocr(self._engine, rectified_card)
        card_size = rectified_card.size

        nik_raw, nik_conf = _row_value(lines, "nik", card_size)
        nama_raw, nama_conf = _row_value(lines, "nama", card_size)
        ttl_raw, ttl_conf = _row_value(lines, "tempat_tanggal_lahir", card_size)
        jk_raw, jk_conf = _row_value(lines, "jenis_kelamin_gol_darah", card_size)
        alamat_raw, alamat_conf = _row_value(lines, "alamat", card_size)
        rtrw_raw, rtrw_conf = _row_value(lines, "rt_rw_kelurahan", card_size)
        kec_raw, kec_conf = _row_value(lines, "kecamatan", card_size)
        agama_raw, agama_conf = _row_value(lines, "agama", card_size)
        status_raw, status_conf = _row_value(lines, "status_perkawinan", card_size)
        pekerjaan_raw, pekerjaan_conf = _row_value(lines, "pekerjaan", card_size)
        wn_raw, wn_conf = _row_value(lines, "kewarganegaraan", card_size)

        tempat_lahir, tanggal_lahir = (
            split_tempat_tanggal_lahir(ttl_raw) if ttl_raw else (None, None)
        )
        jenis_kelamin, golongan_darah = (
            split_jenis_kelamin_gol_darah(jk_raw) if jk_raw else (None, None)
        )
        rt_rw, kelurahan_desa = split_rt_rw_kelurahan(rtrw_raw) if rtrw_raw else (None, None)

        return KtpFields(
            nik=_field(only_digits(nik_raw) if nik_raw else None, nik_conf),
            nama=_field(nama_raw, nama_conf),
            tempat_lahir=_field(tempat_lahir, ttl_conf),
            tanggal_lahir=_field(tanggal_lahir, ttl_conf),
            jenis_kelamin=_field(
                snap_to_enum(jenis_kelamin, GENDERS) if jenis_kelamin else None, jk_conf
            ),
            golongan_darah=_field(golongan_darah, jk_conf),
            alamat=_field(alamat_raw, alamat_conf),
            rt_rw=_field(rt_rw, rtrw_conf),
            kelurahan_desa=_field(kelurahan_desa, rtrw_conf),
            kecamatan=_field(kec_raw, kec_conf),
            agama=_field(snap_to_enum(agama_raw, RELIGIONS) if agama_raw else None, agama_conf),
            status_perkawinan=_field(
                snap_to_enum(status_raw, MARITAL_STATUSES) if status_raw else None, status_conf
            ),
            pekerjaan=_field(
                snap_to_enum(pekerjaan_raw, JOBS) if pekerjaan_raw else None, pekerjaan_conf
            ),
            kewarganegaraan=_field(wn_raw, wn_conf),
            berlaku_hingga=_field_from_match(find_berlaku_hingga(lines)),
            provinsi=_field_from_match(find_provinsi(lines)),
            kota_kabupaten=_field_from_match(find_kota_kabupaten(lines)),
        )
