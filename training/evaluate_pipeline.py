"""Batch-evaluates the real detection+OCR pipeline (the same
KtpExtractionPipeline services/api runs, not a reimplementation) against a
ground-truth-labeled synthetic evaluation set, producing real per-field
accuracy numbers instead of one-off eyeballed checks.

Usage:
    python evaluate_pipeline.py --eval-dir ./eval_set --weights ./runs/ktp_detector/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the actual production pipeline rather than reimplementing the
# detect -> OCR -> assemble-fields glue here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

from app.pipeline import KtpExtractionPipeline  # noqa: E402

from ktp_detection import YoloDetectionService  # noqa: E402
from ktp_ocr import PaddleOcrService  # noqa: E402
from ktp_schema import KtpFields  # noqa: E402

FIELD_NAMES = list(KtpFields.model_fields.keys())


def normalize(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.upper().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the KTP extraction pipeline against ground truth")
    parser.add_argument("--eval-dir", default="./eval_set")
    parser.add_argument("--weights", default="./runs/ktp_detector/weights/best.pt")
    parser.add_argument("--report", default="./eval_report.json")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    image_paths = sorted(eval_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {eval_dir}")
        return

    pipeline = KtpExtractionPipeline(
        detection_service=YoloDetectionService(args.weights),
        ocr_service=PaddleOcrService(),
    )

    correct_counts = {f: 0 for f in FIELD_NAMES}
    total_counts = {f: 0 for f in FIELD_NAMES}
    status_counts: dict[str, int] = {}
    per_image_results = []

    for i, img_path in enumerate(image_paths):
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            continue
        ground_truth = json.loads(gt_path.read_text())

        result = pipeline.run(img_path.read_bytes())
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1

        image_result: dict = {"image": img_path.name, "status": result.status.value, "fields": {}}

        if result.fields is not None:
            for field in FIELD_NAMES:
                gt_value = normalize(ground_truth.get(field))
                if gt_value is None:
                    continue
                total_counts[field] += 1
                extracted = normalize(getattr(result.fields, field).value)
                is_correct = extracted == gt_value
                if is_correct:
                    correct_counts[field] += 1
                image_result["fields"][field] = {
                    "ground_truth": gt_value,
                    "extracted": extracted,
                    "correct": is_correct,
                }

        per_image_results.append(image_result)
        print(f"  {i + 1}/{len(image_paths)}: {img_path.name} -> {result.status.value}")

    print(f"\n{'Field':20s} {'Accuracy':>10s}   (n)")
    print("-" * 45)
    for field in FIELD_NAMES:
        n = total_counts[field]
        if n == 0:
            continue
        accuracy = correct_counts[field] / n
        print(f"{field:20s} {accuracy:9.1%}   ({n})")

    print(f"\nStatus breakdown: {status_counts}")

    Path(args.report).write_text(json.dumps(per_image_results, indent=2))
    print(f"\nFull per-image, per-field report saved to {args.report}")


if __name__ == "__main__":
    main()
