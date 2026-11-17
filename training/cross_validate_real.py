"""4-fold cross-validation of training the detector on the consented real KTP
inputs, on top of the production synthetic recipe.

Every real input is tested once, by a model that never saw it: the labelled
inputs (from prepare_real_labelling.py) are split into 4 folds, pages of one
PDF staying together. Each fold trains on the synthetic set plus the other
three folds' real images, repeated so they aren't drowned out, and is scored
on its held-out fold with the same YoloDetectionService the API uses. The
production model is scored on the same images for comparison.

The synthetic set is generated once and shared by all folds. A fold whose
weights already exist is not retrained, so an interrupted run can be resumed.

Privacy: everything is written under --cv-dir, which must be outside OneDrive
and the repository. That includes the weights, which are derived personal
data once real cards are in the training set. Prints counts only.

Usage:
    python cross_validate_real.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent

FOLDS = 4
REAL_REPEATS = 10  # copies of each real training image, next to ~720 synthetic ones
SYNTHETIC_COUNT = 300  # production recipe: --count 300 --aug-factor 2, default negatives
EPOCHS = 40
BATCH = 16
SEED = 0
IOU_MATCH = 0.5
MAX_DIMENSION = 1600  # same cap as services/api/app/pipeline.py


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _run(command: list[str], log) -> None:
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)


def make_folds(groups: dict[str, int]) -> list[list[str]]:
    """Deals whole groups (one input file each) round-robin after a seeded
    shuffle, so pages of one PDF share a fold."""
    group_ids = sorted(set(groups.values()))
    random.Random(SEED).shuffle(group_ids)
    fold_of = {group: i % FOLDS for i, group in enumerate(group_ids)}
    folds: list[list[str]] = [[] for _ in range(FOLDS)]
    for stem in sorted(groups):
        folds[fold_of[groups[stem]]].append(stem)
    return folds


def build_fold_dataset(
    fold_dir: Path, train_stems: list[str], labelling: Path, base: Path, names: list[str]
) -> Path:
    images_dir = fold_dir / "train" / "images"
    labels_dir = fold_dir / "train" / "labels"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in train_stems:
        for repeat in range(REAL_REPEATS):
            shutil.copy(labelling / "images" / f"{stem}.jpg", images_dir / f"real_{stem}_r{repeat}.jpg")
            shutil.copy(labelling / "labels" / f"{stem}.txt", labels_dir / f"real_{stem}_r{repeat}.txt")

    yaml_path = fold_dir / "ktp.yaml"
    train_dirs = [_posix(base / "train" / "images"), _posix(images_dir)]
    yaml_path.write_text(
        f"train: {json.dumps(train_dirs)}\n"
        f"val: {_posix(base / 'val' / 'images')}\n"
        f"nc: {len(names)}\nnames: [{', '.join(names)}]\n"
    )
    return yaml_path


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(weights: Path, stems: list[str], labelling: Path) -> dict[str, dict]:
    """Per image: whether the service reports a card, and whether its box
    matches the label (IoU >= 0.5). `front` is False for an image labelled as
    showing no KTP front, where a detection is a false positive."""
    from ktp_detection import YoloDetectionService
    from ktp_interfaces import MultipleCardsDetectedError

    service = YoloDetectionService(str(weights))
    results = {}
    for stem in stems:
        image = Image.open(labelling / "images" / f"{stem}.jpg").convert("RGB")
        label_lines = (labelling / "labels" / f"{stem}.txt").read_text().split("\n")
        truth = None
        if label_lines[0].strip():
            _, cx, cy, w, h = (float(v) for v in label_lines[0].split())
            truth = (
                (cx - w / 2) * image.width,
                (cy - h / 2) * image.height,
                (cx + w / 2) * image.width,
                (cy + h / 2) * image.height,
            )
        try:
            detection = service.detect_and_rectify(image)
        except MultipleCardsDetectedError:
            detection = None
        detected = detection is not None
        matched = False
        if detected and truth is not None:
            box = detection[1]
            matched = _iou((box.x1, box.y1, box.x2, box.y2), truth) >= IOU_MATCH
        results[stem] = {"front": truth is not None, "detected": detected, "matched": matched}
    return results


def _pipeline_image(path: Path) -> Image.Image:
    """Same preparation as the API: EXIF rotation, RGB, longest side capped."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    longest = max(image.size)
    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def unseen_midv(midv_root: Path, train_docs: int) -> dict[str, list[Path]]:
    """MIDV documents used for neither training nor validation."""
    from prepare_midv import MIDV_SUBSETS, VAL_FROM_INDEX

    return {
        subset: [
            path
            for path in sorted((midv_root / subset / "images").rglob("*.jpg"))
            if train_docs <= int(path.stem) < VAL_FROM_INDEX
        ]
        for subset in MIDV_SUBSETS
    }


def count_false_positives(weights: Path, image_sets: dict[str, list[Path]]) -> str:
    """How often the service reports a KTP on images of other documents."""
    from ktp_detection import YoloDetectionService
    from ktp_interfaces import MultipleCardsDetectedError

    service = YoloDetectionService(str(weights))
    parts = []
    for name, paths in image_sets.items():
        reported = 0
        for path in paths:
            try:
                reported += service.detect_and_rectify(_pipeline_image(path)) is not None
            except MultipleCardsDetectedError:
                pass
        parts.append(f"{name} {reported}/{len(paths)}")
    return "; ".join(parts)


def summarize(name: str, results: dict[str, dict]) -> str:
    fronts = [r for r in results.values() if r["front"]]
    others = [r for r in results.values() if not r["front"]]
    return (
        f"{name}: detected {sum(r['detected'] for r in fronts)}/{len(fronts)} fronts, "
        f"box correct {sum(r['matched'] for r in fronts)}/{len(fronts)}, "
        f"false positives {sum(r['detected'] for r in others)}/{len(others)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelling", default=r"C:\ktp_sample\labelling")
    parser.add_argument("--cv-dir", default=r"C:\ktp_sample\cv")
    parser.add_argument("--production", default=str(HERE / "runs" / "ktp_detector" / "weights" / "best.pt"))
    parser.add_argument(
        "--midv-root",
        default=None,
        help="Also train on MIDV-2020 mock foreign IDs as a second class, other_document "
        "(see prepare_midv.py), and count false positives on MIDV documents left out of training.",
    )
    parser.add_argument("--midv-train-docs", type=int, default=35, help="MIDV documents per type to train on")
    args = parser.parse_args()
    tag = "midv_" if args.midv_root else ""
    names = ["ktp", "other_document"] if args.midv_root else ["ktp"]

    labelling = Path(args.labelling)
    cv_dir = Path(args.cv_dir).resolve()
    repo = HERE.parent
    if "onedrive" in str(cv_dir).lower() or repo in cv_dir.parents or cv_dir == repo:
        sys.exit("Refusing: --cv-dir must be outside OneDrive and the repository.")
    cv_dir.mkdir(parents=True, exist_ok=True)

    groups = json.loads((labelling / "groups.json").read_text())
    folds = make_folds(groups)
    print(f"Real inputs: {len(groups)} in {FOLDS} folds of {[len(f) for f in folds]}", flush=True)

    with open(cv_dir / "train.log", "a", encoding="utf-8", errors="replace") as log:
        synthetic = cv_dir / "synthetic"
        if not (synthetic / "ktp.yaml").exists():
            print("Generating the synthetic set (once, shared by all folds)...", flush=True)
            _run(
                [
                    sys.executable, "ktp_dataset_generator.py", "--count", str(SYNTHETIC_COUNT),
                    "--output", str(synthetic), "--aug-factor", "2", "--seed", str(SEED),
                ],
                log,
            )

        base = synthetic
        if args.midv_root:
            base = cv_dir / "synthetic_midv"
            if not (base / "ktp.yaml").exists():
                print("Adding MIDV documents to the synthetic set (once)...", flush=True)
                _run(
                    [
                        sys.executable, "prepare_midv.py", "--midv-root", args.midv_root,
                        "--synthetic", str(synthetic), "--output", str(base),
                        "--train-docs-per-type", str(args.midv_train_docs),
                    ],
                    log,
                )

        all_results: dict[str, dict] = {}
        last_results: dict[str, dict] = {}
        for k, held_out in enumerate(folds, start=1):
            run_dir = cv_dir / "runs" / f"{tag}fold{k}"
            best = run_dir / "weights" / "best.pt"
            if not (best.exists() and (run_dir / "results.png").exists()):
                train_stems = [s for i, fold in enumerate(folds) if i != k - 1 for s in fold]
                yaml_path = build_fold_dataset(cv_dir / f"{tag}fold{k}", train_stems, labelling, base, names)
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                started = time.time()
                print(f"Fold {k}: training on {len(train_stems)} real + synthetic...", flush=True)
                _run(
                    [
                        sys.executable, "train_detector.py", "--data", str(yaml_path),
                        "--epochs", str(EPOCHS), "--batch", str(BATCH),
                        "--project", str(cv_dir / "runs"), "--name", f"{tag}fold{k}",
                    ],
                    log,
                )
                print(f"Fold {k}: trained in {(time.time() - started) / 60:.0f} min", flush=True)
            fold_results = score(best, held_out, labelling)
            all_results.update(fold_results)
            last_results.update(score(run_dir / "weights" / "last.pt", held_out, labelling))
            print("  " + summarize(f"fold {k} held-out", fold_results), flush=True)

    production = score(Path(args.production), sorted(groups), labelling)
    print()
    print(summarize("Cross-validated (best.pt)", all_results))
    print(summarize("Cross-validated (last.pt)", last_results))
    print(summarize("Production", production))
    gained = sum(1 for s, r in all_results.items() if r["front"] and r["detected"] and not production[s]["detected"])
    lost = sum(1 for s, r in all_results.items() if r["front"] and not r["detected"] and production[s]["detected"])
    print(f"Fronts gained over production: {gained}; lost: {lost}")

    (cv_dir / f"{tag}results.json").write_text(
        json.dumps({"folds": folds, "cross_validated": all_results, "last": last_results, "production": production}, indent=1)
    )
    print(f"Per-position results: {cv_dir / f'{tag}results.json'}")

    if args.midv_root:
        image_sets = unseen_midv(Path(args.midv_root), args.midv_train_docs)
        print("\nReported as a KTP on unseen MIDV documents (none are KTPs):")
        print(f"  production: {count_false_positives(Path(args.production), image_sets)}", flush=True)
        for k in range(1, FOLDS + 1):
            weights = cv_dir / "runs" / f"{tag}fold{k}" / "weights" / "best.pt"
            print(f"  fold {k}: {count_false_positives(weights, image_sets)}", flush=True)


if __name__ == "__main__":
    main()
