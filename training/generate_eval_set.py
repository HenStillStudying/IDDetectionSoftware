"""Generates a small evaluation set: synthetic KTP scene images, each paired
with a ground-truth JSON of the exact field values printed on the card, for
measuring real pipeline accuracy instead of eyeballing rendered images one
at a time.

Produces base scenes only (rotated card composited onto a random
background, no Albumentations augmentation) — the same difficulty tier as
the training set's un-augmented images.

Usage:
    python generate_eval_set.py --count 30 --output ./eval_set
"""

import argparse
import json
from pathlib import Path

from ktp_dataset_generator import compose_scene, render_ktp


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ground-truth-labeled evaluation set")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", default="./eval_set")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        ktp, ground_truth = render_ktp()
        scene, _bbox = compose_scene(ktp)

        stem = f"eval_{i:04d}"
        scene.save(out_dir / f"{stem}.jpg", quality=90)
        (out_dir / f"{stem}.json").write_text(json.dumps(ground_truth, indent=2))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.count} done...")

    print(f"Done. {args.count} eval images written to {out_dir}")


if __name__ == "__main__":
    main()
