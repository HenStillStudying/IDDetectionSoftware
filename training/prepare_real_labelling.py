"""Prepares (and later imports from) the offline box-labelling page for the
consented real KTP inputs.

Prepare (default): renders every input to a working JPEG exactly as the
pipeline would see it (EXIF-rotated, RGB, longest side capped; PDF pages
rendered in memory), suggests boxes from the production detector, and writes
a folder holding label.html, prefill.js and images/ to open from disk.

Import (--import-labels): turns the page's exported ktp_labels.json into YOLO
label files next to the working images.

Privacy: the working folder holds real personal data, so it must stay outside
OneDrive and the repo, and be deleted after training. Working images are named
by alphabetical position, never by their source filename, and this script
prints counts only. Labels hold box coordinates only.

Usage:
    python prepare_real_labelling.py
    python prepare_real_labelling.py --import-labels %USERPROFILE%\\Downloads\\ktp_labels.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
PAGE_SOURCE = HERE / "label_real_ktp.html"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_DIMENSION = 1600  # same cap as services/api/app/pipeline.py (MAX_UPLOAD_DIMENSION)
PDF_RENDER_SCALE = 200 / 72  # 200 dpi
CONFIDENCE_THRESHOLD = 0.4  # same as YoloDetectionService
JPEG_QUALITY = 92


def _to_pipeline_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    longest = max(image.size)
    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def _render_inputs(source: Path):
    """Yields (stem, group, image). `group` is the input's alphabetical
    position: pages of one PDF share it, so they stay in the same fold."""
    paths = sorted(
        (p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}),
        key=lambda p: p.name.lower(),
    )
    for position, path in enumerate(paths, start=1):
        try:
            if path.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(str(path))
                for page_number in range(len(pdf)):
                    page = pdf[page_number].render(scale=PDF_RENDER_SCALE).to_pil()
                    yield f"{position:02d}_p{page_number + 1}", position, _to_pipeline_image(page)
                pdf.close()
            else:
                with Image.open(path) as image:
                    image.load()
                    yield f"{position:02d}", position, _to_pipeline_image(image)
        except Exception as error:  # noqa: BLE001 - report by position only, never by filename
            print(f"  input {position}: could not be read ({type(error).__name__}), skipped")


def prepare(source: Path, work: Path, weights: Path) -> None:
    from ultralytics import YOLO

    images_dir = work / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    model = YOLO(str(weights))
    ktp_classes = (
        set(model.names) if len(model.names) == 1 else {i for i, n in model.names.items() if n == "ktp"}
    )

    entries = []
    for stem, group, image in _render_inputs(source):
        result = model.predict(image, verbose=False)[0]
        boxes = [
            [round(v, 1) for v in box.xyxy[0].tolist()]
            for box in result.boxes
            if float(box.conf[0]) >= CONFIDENCE_THRESHOLD and int(box.cls[0]) in ktp_classes
        ]
        image.save(images_dir / f"{stem}.jpg", quality=JPEG_QUALITY)
        entries.append(
            {"file": f"{stem}.jpg", "w": image.width, "h": image.height, "group": group, "boxes": boxes}
        )

    (work / "prefill.js").write_text(
        "window.KTP_LABEL_DATA = " + json.dumps({"images": entries}) + ";\n", encoding="utf-8"
    )
    shutil.copy(PAGE_SOURCE, work / "label.html")

    pdf_pages = sum(1 for e in entries if "_p" in e["file"])
    suggested = sum(1 for e in entries if e["boxes"])
    print(f"Inputs: {len(entries)} ({len(entries) - pdf_pages} photos, {pdf_pages} PDF pages)")
    print(f"With a suggested box: {suggested}; to draw by hand: {len(entries) - suggested}")
    print(f"Open: {work / 'label.html'}")


def import_labels(export_path: Path, work: Path, negatives: set[str]) -> None:
    """`negatives` are image stems that show no KTP front (e.g. the back of a
    card). They get an empty label file, which YOLO reads as "nothing to
    detect here", instead of being left out."""
    data = json.loads(export_path.read_text(encoding="utf-8"))
    labels_dir = work / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True)

    unknown = negatives - {Path(entry["file"]).stem for entry in data["images"]}
    if unknown:
        sys.exit(f"Unknown --negative image(s): {', '.join(sorted(unknown))}")

    written = skipped = empty = 0
    groups = {}
    for entry in data["images"]:
        stem = Path(entry["file"]).stem
        if stem in negatives:
            if entry["boxes"]:
                sys.exit(f"Image {stem} is marked --negative but has a box.")
            (labels_dir / f"{stem}.txt").write_text("")
            groups[stem] = entry["group"]
            empty += 1
            continue
        if not entry["reviewed"] or not entry["boxes"] or not (work / "images" / entry["file"]).exists():
            skipped += 1
            continue
        width, height = entry["w"], entry["h"]
        lines = []
        for x1, y1, x2, y2 in entry["boxes"]:
            lines.append(
                f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        groups[stem] = entry["group"]
        written += 1

    (work / "groups.json").write_text(json.dumps(groups, indent=1))
    multi = sum(1 for e in data["images"] if len(e["boxes"]) > 1)
    print(f"Label files written: {written} with a card, {empty} negative; left out (not confirmed): {skipped}")
    if multi:
        print(f"Warning: {multi} image(s) have more than one box.")
    print(f"Labels: {labels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=r"C:\ktp_sample\sample-ktp")
    parser.add_argument("--work", default=r"C:\ktp_sample\labelling")
    parser.add_argument("--weights", default=str(HERE / "runs" / "ktp_detector" / "weights" / "best.pt"))
    parser.add_argument("--import-labels", default=None, metavar="KTP_LABELS_JSON")
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        metavar="STEM",
        help="With --import-labels: an image that shows no KTP front (e.g. 06_p2). Repeatable.",
    )
    args = parser.parse_args()

    work = Path(args.work).resolve()
    repo = HERE.parent
    if "onedrive" in str(work).lower() or repo in work.parents or work == repo:
        sys.exit("Refusing: the working folder must be outside OneDrive and the repository.")

    if args.import_labels:
        import_labels(Path(args.import_labels), work, set(args.negative))
    else:
        prepare(Path(args.source), work, Path(args.weights))


if __name__ == "__main__":
    main()
