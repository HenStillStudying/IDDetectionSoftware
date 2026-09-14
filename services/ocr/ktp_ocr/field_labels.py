"""Fuzzy label matching + geometric value lookup for the fixed KTP layout.

We don't try to read the card generally — we search OCR'd text lines for a
small set of known Indonesian KTP field labels (tolerating OCR noise via
fuzzy string matching), then find each label's value by looking for the
nearest text line positioned below it. The generator (and real KTPs) stack
label above value rather than placing them side by side, and list order
from the OCR engine isn't reliably top-to-bottom, so geometry — not list
position — is what locates the value.

Some visual rows carry two schema fields (e.g. "Jenis Kelamin" also carries
blood type on the same line) — splitting those combined values into
individual KtpFields happens in parsing.py, not here.
"""

from __future__ import annotations

import difflib
import re

from .text_lines import TextLine

LABEL_MATCH_THRESHOLD = 0.6
MAX_VALUE_VERTICAL_GAP_RATIO = 0.08  # fraction of card height
# Labels and values share the same left margin in this layout, so a true
# value's x1 sits within a few pixels of its label's x1. This must stay
# tight — the photo placeholder's "FOTO" caption sits at nearly the same
# height as some field rows, and only the x-shift filter tells them apart.
MAX_VALUE_X_SHIFT_RATIO = 0.04  # fraction of card width

# One entry per visual row on the card. Some rows carry two schema fields —
# splitting those combined values happens in parsing.py.
ROW_LABELS: dict[str, list[str]] = {
    "nik": ["NIK"],
    "nama": ["Nama"],
    "tempat_tanggal_lahir": ["Tempat/Tgl Lahir", "Tempat Tgl Lahir"],
    "jenis_kelamin_gol_darah": ["Jenis Kelamin"],
    "alamat": ["Alamat"],
    "rt_rw_kelurahan": ["RT/RW"],
    "kecamatan": ["Kecamatan"],
    "agama": ["Agama"],
    "status_perkawinan": ["Status Perkawinan"],
    "pekerjaan": ["Pekerjaan"],
    "kewarganegaraan": ["Kewarganegaraan"],
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def find_label_line(lines: list[TextLine], canonical_labels: list[str]) -> TextLine | None:
    best_line: TextLine | None = None
    best_score = 0.0
    for line in lines:
        normalized_line = _normalize(line.text)
        if not normalized_line:
            continue
        for label in canonical_labels:
            score = difflib.SequenceMatcher(None, normalized_line, _normalize(label)).ratio()
            if score > best_score:
                best_score = score
                best_line = line
    return best_line if best_score >= LABEL_MATCH_THRESHOLD else None


def find_value_line(
    lines: list[TextLine], label_line: TextLine, card_size: tuple[int, int]
) -> TextLine | None:
    """The value is the nearest line at or below the label, roughly
    left-aligned with it. Thresholds scale with card size since the
    rectified crop's resolution varies with the source photo.

    "At or below" (not strictly below): OCR sometimes gives the label and
    its value the same y1 when they're a short label next to a taller value
    box, so requiring a strictly positive gap would incorrectly skip the
    real value and match the next field's label instead.
    """
    width, height = card_size
    max_vertical_gap = height * MAX_VALUE_VERTICAL_GAP_RATIO
    max_x_shift = width * MAX_VALUE_X_SHIFT_RATIO

    candidates = [
        line
        for line in lines
        if line is not label_line
        and 0 <= (line.y1 - label_line.y1) <= max_vertical_gap
        and abs(line.x1 - label_line.x1) <= max_x_shift
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda line: (line.y1 - label_line.y1, abs(line.x1 - label_line.x1)),
    )
