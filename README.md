# Bangladeshi Taka Note Detection API (YOLOv11 + FastAPI + Docker)

A deployment project for serving custom Phase-1 YOLOv11 Bangladeshi Taka note detection weights through a REST API and Docker.

> **Required before final submission:** copy your trained Phase-1 weights to `models/best.pt`, add at least five real test images, and capture actual prediction/Docker screenshots. Real denomination detections cannot be reproduced without the trained weights.

## Features

- Loads custom Ultralytics YOLOv11 `.pt` weights.
- Single-image JPEG/PNG inference.
- Returns denomination, confidence, and `xyxy` bounding-box coordinates.
- FastAPI endpoint: `POST /predict`.
- Input validation and meaningful HTTP status codes.
- Docker and Docker Compose support.
- Single-image CLI and five-image API testing scripts.
- Unit tests for API response format and input validation.
- Swagger documentation at `/docs`.

## Project structure

```text
app/                    FastAPI and inference code
scripts/                Single-image and five-image test scripts
tests/                  API contract tests
models/                 Put Phase-1 best.pt here
sample_images/          Put at least five real test images here
docs/                   Evidence instructions and sample response
Dockerfile              Container definition
docker-compose.yml      Optional Compose configuration
requirements.txt        Runtime dependencies
requirements-dev.txt    Test dependencies
```

## Add the trained model

Copy the Phase-1 weight file to:

```text
models/best.pt
```

The API returns the class names embedded in the trained model.

## Local setup (Windows PowerShell)

```powershell
cd path\to\bangladeshi-taka-note-detection-api
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` and `http://localhost:8000/health`.

## Single-image inference

```powershell
python scripts\infer_single.py sample_images\note1.jpg --model models\best.pt --output outputs\note1_result.jpg
```

## API usage

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "accept: application/json" `
  -F "file=@sample_images/note1.jpg;type=image/jpeg"
```

Example response structure:

```json
{
  "filename": "500_taka.jpg",
  "image_width": 1280,
  "image_height": 720,
  "detection_count": 1,
  "inference_time_ms": 48.76,
  "detections": [
    {
      "class_id": 6,
      "denomination": "500 Taka",
      "confidence": 0.9472,
      "bounding_box": {"x1": 214.36, "y1": 132.18, "x2": 1014.72, "y2": 604.91}
    }
  ]
}
```

The numbers above demonstrate the response format only.

## Test at least five images

```powershell
python scripts\test_five_images.py sample_images --url http://localhost:8000/predict
```

The report is written to `docs/test_results/api_test_results.json`.

## Automated API tests

```powershell
pytest -q
```

These tests use a deterministic fake detector to verify the API contract and error handling; they do not measure trained-model accuracy.

## Docker

```powershell
docker build -t taka-note-api:1.0 .
docker run --rm --name taka-note-api -p 8000:8000 taka-note-api:1.0
```

Test from another PowerShell window:

```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST "http://localhost:8000/predict" -F "file=@sample_images/note1.jpg;type=image/jpeg"
```

Docker Compose alternative:

```powershell
docker compose up --build
```

## HTTP status handling

| Situation | Status |
|---|---:|
| Successful prediction | 200 |
| Missing multipart file | 422 |
| Unsupported extension/MIME type | 400 |
| Empty file | 400 |
| File exceeds upload limit | 413 |
| Corrupt JPEG/PNG | 422 |
| Model inference failure | 500 |

## Optional cloud deployment

Create a Docker-based web service on Render, Railway, AWS, Azure, or GCP. Set `MODEL_PATH`, `DEVICE`, and `CONFIDENCE_THRESHOLD`, then test the public `/predict` endpoint. GitHub rejects individual files larger than 100 MB, so use Git LFS or private object storage if the weight file is larger.

## Final submission checklist

- [ ] Add `models/best.pt` from Phase-1.
- [ ] Run the single-image inference script.
- [ ] Save an annotated prediction image.
- [ ] Test five real images with Postman or curl.
- [ ] Complete the accuracy discussion.
- [ ] Build and run the Docker image.
- [ ] Capture actual `/health` and `/predict` screenshots.
- [ ] Add screenshots to the Google Doc.
- [ ] Test the optional public endpoint before claiming bonus marks.

## License note

Review Ultralytics licensing and the license of the dataset/weights before public or commercial deployment.
