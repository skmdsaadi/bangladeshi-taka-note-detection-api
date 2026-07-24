"""FastAPI application for Bangladeshi Taka note detection."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.detector import ModelNotReadyError, TakaNoteDetector
from app.schemas import HealthResponse, PredictionResponse

settings = get_settings()
detector = TakaNoteDetector(
    model_path=settings.model_path,
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
    image_size=settings.image_size,
    device=settings.device,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Attempt eager model loading, while keeping health diagnostics available."""

    try:
        detector.load()
    except ModelNotReadyError:
        pass
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Detect Bangladeshi Taka denominations with custom YOLOv11 weights.",
    lifespan=lifespan,
)


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_: Request, exc: ModelNotReadyError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    return {
        "message": settings.app_name,
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if detector.is_loaded else "model_not_loaded",
        model_loaded=detector.is_loaded,
        model_path=str(settings.model_path),
        device=settings.device,
        version=settings.app_version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(..., description="JPEG or PNG image")) -> PredictionResponse:
    """Accept one image and return all detected notes."""

    allowed_content_types = {"image/jpeg", "image/png"}
    allowed_extensions = (".jpg", ".jpeg", ".png")
    filename = file.filename or "uploaded_image"

    if file.content_type not in allowed_content_types or not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are supported.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.max_upload_mb} MB upload limit.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file is not a valid JPEG or PNG image.",
        ) from exc

    try:
        detections, inference_time_ms = detector.predict(image)
    except ModelNotReadyError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Check the server logs.",
        ) from exc

    width, height = image.size
    return PredictionResponse(
        filename=filename,
        image_width=width,
        image_height=height,
        detection_count=len(detections),
        inference_time_ms=round(inference_time_ms, 2),
        detections=detections,
    )
