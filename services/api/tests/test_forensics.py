import io

from PIL import Image

from app import forensics


def _plain_image() -> Image.Image:
    return Image.new("RGB", (320, 240), color=(180, 180, 180))


def _jpeg_bytes(image: Image.Image, exif_software: str | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs = {}
    if exif_software:
        exif = Image.Exif()
        exif[0x0131] = exif_software
        kwargs["exif"] = exif
    image.save(buf, format="JPEG", quality=90, **kwargs)
    return buf.getvalue()


def test_editor_software_in_exif_is_detected_and_flagged_suspicious():
    data = _jpeg_bytes(_plain_image(), exif_software="Adobe Photoshop 24.0 (Windows)")

    result = forensics.analyze(data)

    assert result.exif_present is True
    assert result.editor_software_detected == "Adobe Photoshop 24.0 (Windows)"
    assert result.suspicious is True
    assert any("Photoshop" in w for w in result.warnings)


def test_missing_exif_is_noted_but_not_suspicious_alone():
    data = _jpeg_bytes(_plain_image())

    result = forensics.analyze(data)

    assert result.exif_present is False
    assert result.editor_software_detected is None
    assert result.suspicious is False
    assert any("no EXIF" in w for w in result.warnings)
