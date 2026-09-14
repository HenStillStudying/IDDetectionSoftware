"""Trains a YOLO detector to locate the KTP card within a scene photo.

Usage:
    python train_detector.py --data ./dataset/ktp.yaml --epochs 30
"""

import argparse
import platform

from ultralytics import YOLO

# On Windows, PyTorch's multiprocess DataLoader workers share tensors via
# memory-mapped files, which reliably fails with "Couldn't open shared file
# mapping" (error 1455) under the default worker count. 0 workers (load data
# on the main process) avoids it; Linux/macOS don't hit this and can use more.
DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the KTP card detector")
    parser.add_argument("--data", default="./dataset/ktp.yaml", help="Path to the ktp.yaml produced by ktp_dataset_generator.py")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--project", default="./runs")
    parser.add_argument("--name", default="ktp_detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
