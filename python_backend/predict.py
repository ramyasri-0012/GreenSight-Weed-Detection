"""Run real weed segmentation and save an annotated image plus JSON predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO


def predict(model_path: Path, image_path: Path, output_dir: Path, confidence: float) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {image_path}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    Image.open(image_path).verify()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    result = model.predict(source=str(image_path), conf=confidence, save=False, verbose=False)[0]
    detections: list[dict[str, Any]] = []
    names = result.names
    polygons = result.masks.xy if result.masks is not None else []
    if result.boxes is not None:
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            class_index = int(class_id)
            detection: dict[str, Any] = {
                "class_id": class_index,
                "class_name": names[class_index],
                "confidence": round(float(score), 4),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
            detection_index = len(detections)
            if detection_index < len(polygons):
                detection["mask_polygon"] = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in polygons[detection_index].tolist()
                ]
            detections.append(detection)

    annotated_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    result.save(filename=str(annotated_path))
    payload = {
        "image": image_path.name,
        "detections": detections,
        "count": len(detections),
        "annotated_image": annotated_path.name,
    }
    (output_dir / f"{image_path.stem}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment weeds in an image")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.image, args.output, args.confidence), indent=2))


if __name__ == "__main__":
    main()
