"""Real OcrService implementation backed by PaddleOCR.

Reads the whole rectified card with a general OCR engine, then locates each
field by fuzzy-matching known Indonesian KTP labels and reading its value —
either from the same OCR line as the label ("Label : Value", confirmed to be
how a real KTP is printed) or, failing that, from the line positioned below
it (how our synthetic generator currently renders every field; see
field_labels.py for why geometry, not OCR line order, locates it). This
tolerates the imperfect rectification produced by YoloDetectionService — a
still-somewhat-rotated or shadow-marked card — far better than a
fixed-pixel-region crop would.
"""

from __future__ import annotations

from PIL.Image import Image as PILImage

from ktp_interfaces import OcrService
from ktp_schema import FieldValue, Gender, JOBS, KtpFields, MARITAL_STATUSES, RELIGIONS

GENDERS = [Gender.MALE.value, Gender.FEMALE.value]
CITIZENSHIPS = ["WNI", "WNA"]  # not in ktp_schema.reference_data — only 2 values, fixed by the KTP form itself

from .field_labels import (
    ROW_LABELS,
    find_label_line,
    find_same_row_value,
    find_value_line,
    merged_label_prefix_value,
    same_line_value,
)
from .parsing import (
    find_berlaku_hingga,
    find_kota_kabupaten,
    find_provinsi,
    normalize_digit_lookalikes,
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

    inline_value = same_line_value(label_line)
    if inline_value is not None:
        return inline_value, label_line.confidence

    merged_value = merged_label_prefix_value(label_line, ROW_LABELS[row_key])
    if merged_value is not None:
        return merged_value, label_line.confidence

    same_row = find_same_row_value(lines, label_line)
    if same_row is not None:
        return same_row

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
        engine: str = "paddle",
        text_detection_model_dir: str | None = None,
        text_recognition_model_dir: str | None = None,
    ):
        # Validate before importing paddleocr — that import alone (not even
        # instantiating a model) takes several seconds via its own heavy
        # dependency chain, which would slow down unit tests exercising
        # just this validation logic for no benefit.
        if engine not in ("paddle", "onnxruntime"):
            raise ValueError(f"Unknown engine {engine!r}; expected 'paddle' or 'onnxruntime'.")
        if engine == "onnxruntime" and not (text_detection_model_dir and text_recognition_model_dir):
            raise ValueError(
                "engine='onnxruntime' requires text_detection_model_dir "
                "and text_recognition_model_dir pointing at pre-converted "
                "ONNX model directories."
            )

        from paddleocr import PaddleOCR

        # Skip doc-orientation/unwarping/textline-orientation: YoloDetectionService
        # already deskews the card upstream, so these three extra model
        # stages would be redundant work on every request.
        kwargs: dict = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
        )

        if engine == "onnxruntime":
            # Same model weights as the default engine, run through ONNX
            # Runtime instead of PaddlePaddle's native inference engine —
            # measured ~3.4x faster on CPU with byte-for-byte identical
            # recognized text on the one test case tried so far (see
            # README; not yet re-measured on this specific deployment's
            # hardware via evaluate_pipeline.py, which is why this isn't
            # the default). Requires model directories already converted
            # to ONNX format (`paddlex --paddle2onnx` — note this
            # conversion step itself has only been confirmed to work on
            # Linux; it fails with a DLL error on Windows, a separate
            # issue from running inference with the already-converted
            # files, which does work on Windows). Doesn't need
            # `enable_mkldnn` below — that's specific to working around a
            # bug in the native Paddle engine, not applicable here.
            kwargs["engine"] = "onnxruntime"
            kwargs["device"] = "cpu"
            kwargs["text_detection_model_dir"] = text_detection_model_dir
            kwargs["text_recognition_model_dir"] = text_recognition_model_dir
        else:
            # enable_mkldnn=False works around a PaddlePaddle/oneDNN crash
            # seen on this stack (NotImplementedError in
            # onednn_instruction.cc) — confirmed not Windows-specific (the
            # same crash reproduces on Linux too, just via a different
            # error message), so this stays regardless of platform;
            # revisit if a paddlepaddle upgrade fixes the underlying bug.
            kwargs["enable_mkldnn"] = False

        self._engine = PaddleOCR(**kwargs)

    def extract_fields(self, rectified_card: PILImage) -> KtpFields:
        lines = run_ocr(self._engine, rectified_card)
        card_size = rectified_card.size

        nik_raw, nik_conf = _row_value(lines, "nik", card_size)
        nama_raw, nama_conf = _row_value(lines, "nama", card_size)
        ttl_raw, ttl_conf = _row_value(lines, "tempat_tanggal_lahir", card_size)
        jk_raw, jk_conf = _row_value(lines, "jenis_kelamin_gol_darah", card_size)
        alamat_raw, alamat_conf = _row_value(lines, "alamat", card_size)
        rt_rw, rtrw_conf = _row_value(lines, "rt_rw", card_size)
        kelurahan_desa, kel_conf = _row_value(lines, "kelurahan_desa", card_size)
        kec_raw, kec_conf = _row_value(lines, "kecamatan", card_size)
        agama_raw, agama_conf = _row_value(lines, "agama", card_size)
        status_raw, status_conf = _row_value(lines, "status_perkawinan", card_size)
        pekerjaan_raw, pekerjaan_conf = _row_value(lines, "pekerjaan", card_size)
        wn_raw, wn_conf = _row_value(lines, "kewarganegaraan", card_size)

        tempat_lahir, tanggal_lahir = (
            split_tempat_tanggal_lahir(ttl_raw) if ttl_raw else (None, None)
        )
        # A date is digit-only by definition, so once it's isolated from
        # tempat_lahir's free text, any letter OCR mistook for a digit
        # (e.g. "13-03-2OO7") is safe to correct back.
        if tanggal_lahir is not None:
            tanggal_lahir = normalize_digit_lookalikes(tanggal_lahir)
        jenis_kelamin, golongan_darah = (
            split_jenis_kelamin_gol_darah(jk_raw) if jk_raw else (None, None)
        )

        # RT/RW and Kel/Desa are separate rows (confirmed against a real
        # KTP), so each is looked up directly above — but if OCR still
        # merges them onto one line/row despite that (older card designs,
        # OCR noise, tight spacing), fall back to splitting them out of
        # the combined rt_rw value the way a single merged row would read.
        if kelurahan_desa is None and rt_rw:
            split_rt_rw, split_kelurahan = split_rt_rw_kelurahan(rt_rw)
            if split_kelurahan is not None:
                rt_rw, kelurahan_desa = split_rt_rw, split_kelurahan

        # RT/RW is digit-only (aside from its "/" separator), so any letter
        # OCR mistook for a digit is safe to correct back.
        if rt_rw is not None:
            rt_rw = normalize_digit_lookalikes(rt_rw)

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
            kelurahan_desa=_field(kelurahan_desa, kel_conf if kel_conf else rtrw_conf),
            kecamatan=_field(kec_raw, kec_conf),
            agama=_field(snap_to_enum(agama_raw, RELIGIONS) if agama_raw else None, agama_conf),
            status_perkawinan=_field(
                snap_to_enum(status_raw, MARITAL_STATUSES) if status_raw else None, status_conf
            ),
            pekerjaan=_field(
                snap_to_enum(pekerjaan_raw, JOBS) if pekerjaan_raw else None, pekerjaan_conf
            ),
            kewarganegaraan=_field(
                snap_to_enum(wn_raw, CITIZENSHIPS) if wn_raw else None, wn_conf
            ),
            berlaku_hingga=_field_from_match(find_berlaku_hingga(lines, card_size)),
            provinsi=_field_from_match(find_provinsi(lines)),
            kota_kabupaten=_field_from_match(find_kota_kabupaten(lines)),
        )
