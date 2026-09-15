"""Fuzzy label matching + value lookup for the fixed KTP layout.

We don't try to read the card generally — we search OCR'd text lines for a
small set of known Indonesian KTP field labels (tolerating OCR noise via
fuzzy string matching), then locate each value using whichever of three
patterns the card's OCR output actually shows, tried in this order:

1. Same OCR line as the label: "NIK : 9205031303070001" as one continuous
   line. Rare in practice on a real card (PaddleOCR usually doesn't merge
   label and value this way), but happens when they sit close enough
   together — see same_line_value.
2. Same row, separate OCR box: label and value detected as two distinct
   boxes with overlapping y-ranges, value to the right — confirmed to be
   the *dominant* pattern on a real KTP (the government form's column
   spacing is wide enough that OCR splits them rather than merging or
   stacking them). A row can have multiple boxes to the right (e.g. "Jenis
   Kelamin" also carries gender, "Gol. Darah", and the blood type letter as
   three separate boxes) — see find_same_row_value.
3. Stacked on the line below the label — how our synthetic dataset
   generator currently renders every field (label on one line, ": value" on
   the next). Value lookup here is geometric (nearest line below, not list
   order) since OCR line order isn't reliably top-to-bottom — see
   find_value_line.

All three are needed because the synthetic generator's assumption
(stacked) turned out not to match real KTPs (same-row, separate boxes) — a
real card exposed a systematic one-row-shifted misread across every field
before this fix.

Some visual rows carry two schema fields (e.g. "Jenis Kelamin" also carries
blood type) — splitting those combined values into individual KtpFields
happens in parsing.py, not here.
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
    """Matches against the text before the line's first colon, not the
    whole line. On a stacked-layout line this is the same thing (the label
    has no colon at all). On a same-line "Label : Value" line, comparing
    against the *whole* line (including the value) dilutes the match ratio
    badly for short labels — "nik" vs "nik9205031303070001..." scores far
    below threshold — so only the portion before the colon is compared.
    """
    best_line: TextLine | None = None
    best_score = 0.0
    for line in lines:
        label_portion = line.text.split(":", 1)[0]
        normalized_line = _normalize(label_portion)
        if not normalized_line:
            continue
        for label in canonical_labels:
            score = difflib.SequenceMatcher(None, normalized_line, _normalize(label)).ratio()
            if score > best_score:
                best_score = score
                best_line = line
    return best_line if best_score >= LABEL_MATCH_THRESHOLD else None


def same_line_value(label_line: TextLine) -> str | None:
    """Returns the text after the label line's first colon, if there's
    anything substantial there — the real-KTP case, "Label : Value" as one
    OCR line. Returns None when there's no colon or nothing meaningful
    follows it, which is the synthetic generator's stacked-layout case
    (the label alone, with its value on the line below instead).
    """
    parts = label_line.text.split(":", 1)
    if len(parts) != 2:
        return None
    value = parts[1].strip()
    return value or None


MIN_ROW_OVERLAP_RATIO = 0.3  # fraction of the label's own height its y-range must overlap
MAX_CANDIDATE_HEIGHT_RATIO = 3.0  # reject boxes taller than this multiple of the label's height
MAX_CANDIDATE_GAP_RATIO = 8.0  # stop concatenating once the horizontal gap exceeds this multiple of the label's height


def find_same_row_value(lines: list[TextLine], label_line: TextLine) -> tuple[str, float] | None:
    """The value sits in one or more separate OCR boxes to the right of the
    label, on the same visual row — confirmed to be the dominant pattern on
    a real KTP, where the form's column spacing is wide enough that OCR
    detects label and value as distinct boxes rather than merging them into
    one line or stacking them vertically.

    Concatenates every qualifying box left-to-right (not just the nearest),
    since a row can carry more than one — "Jenis Kelamin" has separate boxes
    for the gender, the literal "Gol. Darah" text, and the blood-type letter.

    Two guards, both found necessary by testing against a real KTP:
    - Uses the label's *left* edge as the cutoff for "to the right", not its
      right edge — a garbled OCR read can inflate a label's own box width
      (e.g. "Status Perkawinan" misread with trailing noise), which would
      otherwise push the cutoff past where the true value box starts and
      wrongly exclude it.
    - Rejects candidates far taller than the label — a KTP's background
      security watermark ("KARTU TANDA PENDUDUK" printed repeatedly as a
      diagonal texture) gets OCR'd as spurious, unusually tall text
      fragments that can otherwise satisfy the row-overlap check and
      contaminate the result.
    """
    label_height = label_line.y2 - label_line.y1
    candidates = []
    for line in lines:
        if line is label_line or line.x1 <= label_line.x1:
            continue
        candidate_height = line.y2 - line.y1
        if label_height <= 0 or candidate_height > label_height * MAX_CANDIDATE_HEIGHT_RATIO:
            continue
        overlap = min(line.y2, label_line.y2) - max(line.y1, label_line.y1)
        if overlap / label_height < MIN_ROW_OVERLAP_RATIO:
            continue
        candidates.append(line)

    if not candidates:
        return None

    candidates.sort(key=lambda line: line.x1)

    # Stop at the first large horizontal gap: legitimate multi-box values
    # (e.g. "Jenis Kelamin"'s gender/"Gol. Darah"/letter boxes) sit within
    # a few hundred pixels of each other, but an unrelated box positioned
    # far to the right (e.g. an issue date near the signature, coincidentally
    # overlapping a field's y-range) can be hundreds of pixels further —
    # scaled by the label's own height as a proxy for this card's text scale.
    max_gap = label_height * MAX_CANDIDATE_GAP_RATIO
    kept = [candidates[0]]
    for line in candidates[1:]:
        if line.x1 - kept[-1].x2 > max_gap:
            break
        kept.append(line)

    value = " ".join(line.text for line in kept).strip()
    value = re.sub(r"^[:\s]+", "", value)  # some rows keep the ':' on the value box, not the label
    if not value:
        return None
    confidence = sum(line.confidence for line in kept) / len(kept)
    return value, confidence


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
