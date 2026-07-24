"""Pydantic response schemas for the API."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(description="Left coordinate in pixels")
    y1: float = Field(description="Top coordinate in pixels")
    x2: float = Field(description="Right coordinate in pixels")
    y2: float = Field(description="Bottom coordinate in pixels")


class Detection(BaseModel):
    class_id: int
    denomination: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox


class PredictionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    inference_time_ms: float
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    version: str
