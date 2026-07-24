"""Run one-image YOLO inference and save an annotated result."""

import argparse
import json
from pathlib import Path

from app.detector import TakaNoteDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.jpg"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    detector = TakaNoteDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    detections = detector.predict_and_save(args.image, args.output)
    print(json.dumps({"detections": detections}, indent=2))
    print(f"Annotated result saved to: {args.output}")


if __name__ == "__main__":
    main()
