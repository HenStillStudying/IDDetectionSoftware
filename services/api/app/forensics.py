"""Tier-1 tamper-detection signals for an uploaded KTP photo.

Only EXIF inspection is active here. Error Level Analysis was built and
tested first, then shelved: tested against one real (unedited) KTP photo,
it returned a false positive — a 4.6 std-dev regional anomaly on a
genuinely untampered card. Root cause, not just a bad threshold: a real
photo's card body, text edges, photo backdrop, and background surface all
carry naturally different amounts of high-frequency detail, which produces
the same kind of regional compression-response variance this check was
looking for from tampering. It also can't distinguish that from the extra
compression generation something like WhatsApp adds when forwarding an
image, a well-documented real-world failure mode of naive ELA. The
synthetic test images used during development (smooth gradients) never
exercised this — the same lesson this project has already learned once
with the detector: a synthetic-only test set can hide a real-photo-only
failure. Revisit if there's a way to isolate ELA to just the card region
(excluding the background) and to test it against more than one real
photo — not by just tuning the threshold on this same single case.

JPEG block-grid misalignment and per-field font-consistency checks were
never built at all — both need real design work to implement reliably
rather than shipping something unvalidated.

This is a warnings/scoring layer, not a verdict. A clean result is not
proof of authenticity, and `suspicious=True` is not proof of forgery — a
sufficiently careful edit (careful recompression, or a print-and-rephoto
attack) can defeat single-image forensics entirely regardless. This exists
to catch obvious/amateur edits, not to hold up against a motivated forger.

Must run on the *original* uploaded bytes, never on anything this pipeline
has already resized or re-encoded — both can strip the EXIF this still
checks. Callers must have already validated that `image_bytes` decodes
cleanly (see pipeline.py) — this module does not re-validate that itself.
"""

from __future__ import annotations

import io

from PIL import Image

from ktp_schema import ForensicsResult

# The EXIF "Software" tag (0x0131) — set by most image editors, though
# trivially absent from a re-exported/flattened/screenshotted file, so this
# is a weak, free signal, not a reliable one.
_SOFTWARE_TAG = 0x0131
_KNOWN_EDITOR_NAMES = ("photoshop", "gimp", "affinity", "pixelmator", "snapseed", "lightroom")


def analyze(image_bytes: bytes) -> ForensicsResult:
    image = Image.open(io.BytesIO(image_bytes))

    exif = image.getexif()
    exif_present = bool(exif)
    editor_software_detected = _detect_editor_software(exif)

    warnings: list[str] = []
    if editor_software_detected:
        warnings.append(f"Image metadata indicates it was processed with {editor_software_detected}.")
    if not exif_present:
        warnings.append(
            "Image has no EXIF metadata — inconclusive on its own (screenshots and "
            "some editors strip it too), but most real camera photos have some."
        )

    return ForensicsResult(
        exif_present=exif_present,
        editor_software_detected=editor_software_detected,
        suspicious=bool(editor_software_detected),
        warnings=warnings,
    )


def _detect_editor_software(exif) -> str | None:
    software = exif.get(_SOFTWARE_TAG)
    if not software:
        return None
    lowered = str(software).lower()
    if any(name in lowered for name in _KNOWN_EDITOR_NAMES):
        return str(software)
    return None
