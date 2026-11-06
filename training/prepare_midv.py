"""Merges real photos and scans of other ID documents (MIDV-2020) into a
synthetic KTP dataset, as a second class, "other_document".

The detector trained on synthetic KTPs alone found 16 of 28 real cards, and
retraining swung between 1 and 17 depending on the random seed: synthetic
scenes don't teach what real phone photos look like. MIDV-2020 has 1000
real photos (low light, glare, outdoors, cluttered backgrounds) and 1000
scans of mock ID documents from 10 countries, with generated faces and
fictional data. They're labelled as a separate class so the detector learns
real-photo conditions without learning that any ID card is a KTP.

MIDV-2020 is licensed CC BY-SA 2.5 (Bulatov et al., "MIDV-2020: A
Comprehensive Benchmark Dataset for Identity Document Analysis", 2021).
Faces were made with Generated Photos (https://generated.photos/).

Usage (after ktp_dataset_generator.py has written --synthetic):
    python prepare_midv.py --midv-root C:/midv_download --synthetic ./dataset --output ./dataset_midv
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

MIDV_SUBSETS = ("photo", "scan_rotated")
OTHER_DOCUMENT_CLASS = 1
MAX_SIDE = 1280  # YOLO trains at 640; full-size phone photos only cost disk
VAL_FROM_INDEX = 85  # per document type, documents 85-99 go to val


def _doc_box(regions: list[dict]) -> tuple[float, float, float, float] | None:
    for region in regions:
        if region["region_attributes"].get("field_name") == "doc_quad":
            shape = region["shape_attributes"]
            xs, ys = shape["all_points_x"], shape["all_points_y"]
            return min(xs), min(ys), max(xs), max(ys)
    return None


def _yolo_line(cls: int, box, width: int, height: int) -> str:
    x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
    return (f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def add_midv(midv_root: Path, output: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for subset in MIDV_SUBSETS:
        for annotation_file in sorted((midv_root / subset / "annotations").glob("*.json")):
            doc_type = annotation_file.stem
            metadata = json.loads(annotation_file.read_text(encoding="utf-8"))["_via_img_metadata"]
            for entry in metadata.values():
                box = _doc_box(entry["regions"])
                if box is None:
                    continue
                split = "val" if int(Path(entry["filename"]).stem) >= VAL_FROM_INDEX else "train"
                with Image.open(midv_root / subset / "images" / doc_type / entry["filename"]) as img:
                    # Half the photos are stored sideways with an EXIF
                    # rotate-90 tag, and their annotations are in the
                    # upright frame.
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    scale = min(1.0, MAX_SIDE / max(img.size))
                    if scale < 1.0:
                        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
                        box = tuple(v * scale for v in box)
                    stem = f"midv_{subset}_{doc_type}_{Path(entry['filename']).stem}"
                    img.save(output / split / "images" / f"{stem}.jpg", quality=90)
                    (output / split / "labels" / f"{stem}.txt").write_text(
                        _yolo_line(OTHER_DOCUMENT_CLASS, box, img.width, img.height))
                counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--midv-root", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path, help="Output of ktp_dataset_generator.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Synthetic KTP labels are already class 0 ("ktp"); copy them as they are.
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            src, dst = args.synthetic / split / kind, args.output / split / kind
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            dst.mkdir(parents=True, exist_ok=True)

    counts = add_midv(args.midv_root, args.output)
    (args.output / "ktp.yaml").write_text(f"""\
path: {args.output.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 2
names: [ktp, other_document]
""")
    print(f"MIDV images added: train {counts['train']}, val {counts['val']}")
    print(f"Config: {args.output / 'ktp.yaml'}")


if __name__ == "__main__":
    main()
