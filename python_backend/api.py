"""FastAPI service for the trained GreenSight weed detector."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from predict import predict

MODEL_PATH = Path(os.getenv("GREENSIGHT_MODEL", "runs/weed_detector/weights/best.pt"))
OUTPUT_DIR = Path(os.getenv("GREENSIGHT_OUTPUT_DIR", "outputs"))
app = FastAPI(title="GreenSight Weed Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5175",
    "https://greensight-weed-detection.vercel.app",
],    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": MODEL_PATH.is_file(), "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict_image(image: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, JPEG, or WEBP image.")
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Trained model not found at {MODEL_PATH}. Run train.py first.",
        )

    suffix = Path(image.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image extension.")
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        return predict(MODEL_PATH, temporary_path, OUTPUT_DIR, confidence=0.25)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary_path.unlink(missing_ok=True)


@app.get("/outputs/{filename}")
def output_file(filename: str) -> FileResponse:
    requested = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in requested.parents or not requested.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(requested)
