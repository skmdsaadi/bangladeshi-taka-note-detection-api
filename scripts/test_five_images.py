"""Send at least five images to the running /predict endpoint."""

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--output", type=Path, default=Path("docs/test_results/api_test_results.json"))
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 5:
        raise ValueError("Add at least five JPEG/PNG images to the test directory.")

    report = []
    with httpx.Client(timeout=120.0) as client:
        for image_path in images[:5]:
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            with image_path.open("rb") as image_file:
                response = client.post(
                    args.url,
                    files={"file": (image_path.name, image_file, mime)},
                )
            result = {
                "image": image_path.name,
                "status_code": response.status_code,
                "response": response.json(),
            }
            report.append(result)
            print(json.dumps(result, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
