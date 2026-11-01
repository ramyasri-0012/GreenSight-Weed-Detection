"""Train a real YOLO weed segmentation model on a YOLO-format dataset.

Expected dataset layout:
dataset/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt

Each segmentation label line is: class_id x1 y1 x2 y2 ... xn yn
with polygon coordinates normalized to 0..1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def validate_dataset(data_file: Path) -> None:
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_file}")
    dataset_root = data_file.parent.parent / "dataset"
    required = (
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Dataset folders are missing. Create these folders before training:\n"
            + "\n".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GreenSight weed segmentation model")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, 0, 1, or auto")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="weed_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = args.data.resolve()
    validate_dataset(data_file)
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must be positive values")

    model = YOLO(args.model)
    train_options = {
        "data": str(data_file),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "patience": 30,
        "plots": True,
        "exist_ok": True,
    }
    if args.device:
        train_options["device"] = args.device
    results = model.train(**train_options)
    print(f"Training complete. Results saved to: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
