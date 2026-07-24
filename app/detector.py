"""YOLOv11 model loading and single-image inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


class ModelNotReadyError(RuntimeError):
    """Raised when the trained weights are unavailable."""


class TakaNoteDetector:
    """Thin wrapper around an Ultralytics YOLO detection model."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model: YOLO | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the custom Phase-1 weights from disk."""

        if not self.model_path.is_file():
            raise ModelNotReadyError(
                f"Model weights were not found at '{self.model_path}'. "
                "Copy the trained Phase-1 weights to models/best.pt or set MODEL_PATH."
            )
        self.model = YOLO(str(self.model_path))

    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], float]:
        """Run detection and return JSON-ready detections and elapsed milliseconds."""

        if self.model is None:
            self.load()

        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image)

        started = time.perf_counter()
        results = self.model.predict(
            source=image_array,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        detections: list[dict[str, Any]] = []
        if not results:
            return detections, elapsed_ms

        result = results[0]
        names = result.names
        if result.boxes is None:
            return detections, elapsed_ms

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "denomination": str(names[class_id]),
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

        return detections, elapsed_ms

    def predict_and_save(self, image_path: Path, output_path: Path) -> list[dict[str, Any]]:
        """Run inference, save an annotated image, and return detections."""

        if self.model is None:
            self.load()

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = results[0].plot()
        Image.fromarray(annotated[..., ::-1]).save(output_path)

        result = results[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_id": class_id,
                        "denomination": str(result.names[class_id]),
                        "confidence": round(float(box.conf.item()), 4),
                        "bounding_box": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    }
                )
        return detections
