"""Trains a YOLO detector to locate the KTP card within a scene photo.

Usage:
    python train_detector.py --data ./dataset/ktp.yaml --epochs 30
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the KTP card detector")
    parser.add_argument("--data", default="./dataset/ktp.yaml", help="Path to the ktp.yaml produced by ktp_dataset_generator.py")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="./runs")
    parser.add_argument("--name", default="ktp_detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
