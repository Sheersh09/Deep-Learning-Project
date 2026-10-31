"""HTTP API for the saved baseline tomato leaf classifier."""

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from .inference import DEFAULT_MODEL_PATH, InvalidImageError, load_baseline_model, predict_image_bytes


APPLICATION_VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PREDICTION_WARNING = (
    "The model was trained on PlantVillage images and may not generalize reliably "
    "to real-world photographs. It is an educational classifier, not a validated "
    "diagnostic tool."
)
logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_version: str


class VersionResponse(BaseModel):
    application_version: str
    model_version: str
    artifact_sha256: str


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.model = None
    application.state.model_version = None
    application.state.artifact_sha256 = None
    try:
        model = load_baseline_model()
        with DEFAULT_MODEL_PATH.open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        application.state.model = model
        application.state.model_version = f"baseline-{digest[:12]}"
        application.state.artifact_sha256 = digest
    except Exception:
        logger.exception("Baseline model unavailable at startup")

    try:
        yield
    finally:
        application.state.model = None
        application.state.model_version = None
        application.state.artifact_sha256 = None


app = FastAPI(title="Tomato Leaf Classifier", version=APPLICATION_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Report process liveness without accessing the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "model", None) is None or not getattr(
        request.app.state, "model_version", None
    ):
        raise HTTPException(status_code=503, detail="Model unavailable.")
    return {"status": "ready"}


@app.get("/version", response_model=VersionResponse)
def version(request: Request) -> VersionResponse:
    model_version = getattr(request.app.state, "model_version", None)
    digest = getattr(request.app.state, "artifact_sha256", None)
    if getattr(request.app.state, "model", None) is None or not model_version or not digest:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    return VersionResponse(
        application_version=APPLICATION_VERSION,
        model_version=model_version,
        artifact_sha256=digest,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    description=PREDICTION_WARNING,
    responses={
        400: {"description": "Invalid image"},
        413: {"description": "Image too large"},
        503: {"description": "Model unavailable"},
        500: {"description": "Inference failed"},
    },
)
def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    model_version = getattr(request.app.state, "model_version", None)
    if model is None or not model_version:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Uploaded image could not be read.") from exc
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit.")

    try:
        result = predict_image_bytes(image_bytes, model)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected inference failure")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictionResponse(**result, model_version=model_version)
