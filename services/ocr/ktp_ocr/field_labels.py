"""Fuzzy label matching + value lookup for the fixed KTP layout.

We don't try to read the card generally — we search OCR'd text lines for a
small set of known Indonesian KTP field labels (tolerating OCR noise via
fuzzy string matching), then locate each value using whichever of four
patterns the card's OCR output actually shows, tried in this order:

1. Same OCR line as the label, colon-separated: "NIK : 9205031303070001"
   as one continuous line. Happens when label and value sit close enough
   together for PaddleOCR to merge their boxes — see same_line_value.
2. Same OCR line as the label, *no* colon at all: "Status Perkawinarc
   BELUM KAWIN" merged into one box with nothing separating them —
   confirmed on a real KTP where PaddleOCR's merging behavior varies even
   for the same physical layout style. Without this, the match falls
   through every other strategy (there's no colon to split on, and
   nothing else qualifies as a same-row box since the value is stuck
   inside the label's own) all the way to the stacked-below fallback,
   which then wrongly grabs the *next row's label* text instead (every
   row's label shares the same left margin) — see
   merged_label_prefix_value.
3. Same row, separate OCR box: label and value detected as two distinct
   boxes with overlapping y-ranges, value to the right — confirmed to be
   the *dominant* pattern on a real KTP (the government form's column
   spacing is wide enough that OCR splits them rather than merging or
   stacking them). A row can have multiple boxes to the right (e.g. "Jenis
   Kelamin" also carries gender, "Gol. Darah", and the blood type letter as
   three separate boxes) — see find_same_row_value.
4. Stacked on the line below the label — how our synthetic dataset
   generator currently renders every field (label on one line, ": value" on
   the next). Value lookup here is geometric (nearest line below, not list
   order) since OCR line order isn't reliably top-to-bottom — see
   find_value_line.

All four are needed because no single assumption holds consistently: the
synthetic generator's original assumption (stacked) didn't match real
KTPs at all (same-row, separate boxes) — a real card exposed a systematic
one-row-shifted misread across every field before that fix — and even
after that fix, the *same* physical row printed on the *same* real card
has been OCR'd as colon-merged, colonless-merged, and separate-box across
different test runs.

Some visual rows carry two schema fields (e.g. "Jenis Kelamin" also carries
blood type) — splitting those combined values into individual KtpFields
happens in parsing.py, not here.
"""

from __future__ import annotations

import difflib
import re

from .text_lines import TextLine

LABEL_MATCH_THRESHOLD = 0.6
# Stricter than LABEL_MATCH_THRESHOLD — see merged_label_prefix_value's
# docstring for why a looser threshold there caused a real regression.
MERGED_LABEL_SPLIT_THRESHOLD = 0.85
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
    "rt_rw": ["RT/RW"],
    "kelurahan_desa": ["Kel/Desa", "Kelurahan/Desa"],
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


def merged_label_prefix_value(label_line: TextLine, canonical_labels: list[str]) -> str | None:
    """Handles a label and its value merged onto one OCR line with *no*
    colon between them at all (e.g. "Status Perkawinarc BELUM KAWIN") — a
    variant `same_line_value` can't handle since it specifically looks for
    a colon. Confirmed necessary against a real KTP: without this, the
    match fell through `same_line_value` (no colon to split on) and
    `find_same_row_value` (nothing else qualifies at that exact row, since
    the value is stuck inside the label's own box) all the way to the
    stacked-below fallback, which then wrongly grabbed the *next row's
    label* text instead (it happens to share the same left margin every
    row uses).

    Finds the word-count prefix of the line that best fuzzy-matches a
    canonical label (the same matching `find_label_line` uses, just
    applied at each possible split point instead of the whole line), and
    returns everything after that prefix as the value. Returns None if the
    line has a colon (that's `same_line_value`'s case) or no split point
    scores high enough to trust.

    Uses a stricter threshold than `find_label_line`'s (0.6): a *partial*
    prefix of a genuine multi-word label can itself already clear 0.6
    against the full canonical label (e.g. "Tempat/Tgl" alone scores 0.78
    against "Tempat/Tgl Lahir") — confirmed to cause a real regression, a
    stacked-layout "Tempat/Tgl Lahir" label-only line getting its own
    trailing word "Lahir" sliced off and mistaken for a value. A genuine
    merged label+value line's true split point scores far higher (~0.91
    for "Status Perkawinarc" against "Status Perkawinan") since the whole
    label, not a fragment of it, precedes the value there.
    """
    if ":" in label_line.text:
        return None

    words = label_line.text.split()
    best_prefix_len: int | None = None
    best_score = 0.0
    for prefix_len in range(1, len(words)):
        normalized_prefix = _normalize(" ".join(words[:prefix_len]))
        if not normalized_prefix:
            continue
        for label in canonical_labels:
            score = difflib.SequenceMatcher(None, normalized_prefix, _normalize(label)).ratio()
            if score > best_score:
                best_score, best_prefix_len = score, prefix_len

    if best_prefix_len is None or best_score < MERGED_LABEL_SPLIT_THRESHOLD:
        return None

    value = " ".join(words[best_prefix_len:]).strip()
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

    Four guards now, the third and fourth found necessary on the synthetic
    generator's tighter row spacing:
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
    - Stops concatenating at the first large horizontal gap (below) — this
      alone already correctly rejects a stray box that happens to overlap a
      label's y-range but sits far away in x (e.g. an issuance date near the
      signature, confirmed against a real KTP).
    - De-duplicates candidates that start at nearly the same x as each
      other *before* that gap-based walk. An all-caps value's OCR-detected
      box reaches full cap-height while its (usually mixed-case) label
      doesn't, so a value's box consistently starts a bit *above* the y its
      label was drawn at — confirmed via raw OCR geometry dumps to be large
      enough, relative to the synthetic generator's row spacing, that a
      label's overlap check alone sometimes also (wrongly) qualifies the
      *next* row's value in the same fixed value column (same x, different
      row) as "Jenis Kelamin" concatenating "Alamat"'s value too (silently
      breaking golongan_darah's end-anchored regex), and "Nama"
      concatenating "Tempat/Tgl Lahir"'s date. Only same-x1 candidates are
      deduplicated this way (kept: whichever is at-or-above the label,
      matching the observed offset direction) — a genuinely stray box like
      the issuance date sits at a very different x and is already excluded
      by the gap-based walk below, untouched by this guard. An earlier
      version of this guard filtered *all* candidates by closeness to a
      single directional anchor rather than just same-column ties, which
      wrongly discarded a real KTP's correct "Berlaku Hingga" value in
      favor of that same issuance date — the gap-based walk was already
      handling that case correctly on its own and didn't need help.
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

    label_center = (label_line.y1 + label_line.y2) / 2

    def _center(line: TextLine) -> float:
        return (line.y1 + line.y2) / 2

    def _row_preference(line: TextLine) -> tuple[int, float]:
        return (0 if _center(line) <= label_center else 1, abs(_center(line) - label_center))

    x_tie_tolerance = label_height * 0.5
    deduped: list[TextLine] = []
    for line in candidates:
        if deduped and line.x1 - deduped[-1].x1 <= x_tie_tolerance:
            if _row_preference(line) < _row_preference(deduped[-1]):
                deduped[-1] = line
            continue
        deduped.append(line)
    candidates = deduped

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
