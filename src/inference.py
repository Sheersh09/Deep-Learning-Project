"""Single-image inference with the saved baseline tomato CNN."""

from io import BytesIO
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

from .model import IMG_SIZE


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "models" / "model_baseline.keras"
MAX_IMAGE_PIXELS = 20_000_000
CLASS_NAMES = (
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
)
IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
}


class InvalidImageError(ValueError):
    """The supplied bytes are not a supported, readable image."""


class PredictionResult(TypedDict):
    predicted_class: str
    probabilities: dict[str, float]


def load_baseline_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load the existing model for reuse across predictions."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    return keras.models.load_model(path, compile=False)


def _preprocess_image_bytes(image_bytes: bytes, expected_format: str | None = None) -> np.ndarray:
    if not image_bytes:
        raise InvalidImageError("Image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image_format = source.format
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image exceeds the decoded-pixel limit.")
            source.verify()
    except InvalidImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Image is unreadable or corrupt.") from exc

    if image_format not in IMAGE_FORMATS.values():
        raise InvalidImageError(f"Unsupported image format: {image_format}.")
    if expected_format is not None and image_format != expected_format:
        raise InvalidImageError("Image content does not match its file extension.")

    try:
        image = load_img(
            BytesIO(image_bytes),
            color_mode="rgb",
            target_size=(IMG_SIZE, IMG_SIZE),
            interpolation="nearest",
            keep_aspect_ratio=False,
        )
        try:
            pixels = img_to_array(image)
        finally:
            image.close()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Image is unreadable or corrupt.") from exc

    # Match the evaluation generator's float array and rescale operation.
    pixels = ImageDataGenerator(rescale=1.0 / 255).standardize(pixels)
    return np.expand_dims(pixels, axis=0)


def predict_image_bytes(image_bytes: bytes, model: keras.Model) -> PredictionResult:
    """Predict from in-memory image bytes, suitable for a future API caller."""
    batch = _preprocess_image_bytes(image_bytes)
    return _predict_batch(batch, model)


def _predict_batch(batch: np.ndarray, model: keras.Model) -> PredictionResult:
    scores = np.asarray(model.predict(batch, verbose=0))
    if scores.shape != (1, len(CLASS_NAMES)) or not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid prediction probabilities.")

    probabilities = {
        class_name: float(scores[0, index])
        for index, class_name in enumerate(CLASS_NAMES)
    }
    predicted_index = int(np.argmax(scores[0]))
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": probabilities,
    }


def predict_image(image_path: str | Path, model: keras.Model) -> PredictionResult:
    """Validate a local image path and predict its tomato leaf class."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    expected_format = IMAGE_FORMATS.get(path.suffix.lower())
    if expected_format is None:
        raise ValueError(f"Unsupported image extension: {path.suffix or '(none)'}")

    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Image cannot be read: {path}") from exc

    batch = _preprocess_image_bytes(image_bytes, expected_format=expected_format)
    return _predict_batch(batch, model)
