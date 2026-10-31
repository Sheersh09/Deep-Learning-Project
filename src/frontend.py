"""Minimal Streamlit client for the tomato leaf prediction API."""

import math
import os
from io import BytesIO
from typing import TypedDict

import requests
import streamlit as st
from PIL import Image


DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PREVIEW_PIXELS = 20_000_000
HTTP_TIMEOUT = (3, 30)


class FrontendError(Exception):
    """An upload or API problem that can be shown to the user."""


class PredictionResponse(TypedDict):
    predicted_class: str
    probabilities: dict[str, float]
    model_version: str


def prepare_preview(image_bytes: bytes) -> Image.Image:
    """Check size and basic readability before displaying an image."""
    if not image_bytes:
        raise FrontendError("The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise FrontendError("The image exceeds the 10 MiB upload limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_PREVIEW_PIXELS:
                raise FrontendError("The image is too large to preview safely.")
            image.load()
            return image.copy()
    except FrontendError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise FrontendError("The image is unreadable or corrupt.") from exc


def validate_prediction_response(payload: object) -> PredictionResponse:
    """Reject malformed API success responses before displaying them."""
    if not isinstance(payload, dict):
        raise FrontendError("The prediction service returned an invalid response.")

    predicted_class = payload.get("predicted_class")
    model_version = payload.get("model_version")
    probabilities = payload.get("probabilities")
    if not isinstance(predicted_class, str) or not predicted_class.strip():
        raise FrontendError("The prediction service returned an invalid response.")
    if not isinstance(model_version, str) or not model_version.strip():
        raise FrontendError("The prediction service returned an invalid response.")
    if not isinstance(probabilities, dict) or len(probabilities) != 3:
        raise FrontendError("The prediction service returned invalid probabilities.")
    if predicted_class not in probabilities:
        raise FrontendError("The prediction service returned invalid probabilities.")

    checked_probabilities: dict[str, float] = {}
    for name, value in probabilities.items():
        if not isinstance(name, str) or not name:
            raise FrontendError("The prediction service returned invalid probabilities.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FrontendError("The prediction service returned invalid probabilities.")
        if not 0.0 <= value <= 1.0:
            raise FrontendError("The prediction service returned invalid probabilities.")
        score = float(value)
        if not math.isfinite(score):
            raise FrontendError("The prediction service returned invalid probabilities.")
        checked_probabilities[name] = score

    if not math.isclose(sum(checked_probabilities.values()), 1.0, rel_tol=0.0, abs_tol=0.001):
        raise FrontendError("The prediction service returned invalid probabilities.")

    return {
        "predicted_class": predicted_class,
        "probabilities": checked_probabilities,
        "model_version": model_version,
    }


def request_prediction(
    image_bytes: bytes, filename: str, content_type: str, backend_url: str
) -> PredictionResponse:
    """Send the unchanged image bytes to the existing API."""
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise FrontendError("The image exceeds the 10 MiB upload limit.")
    base_url = backend_url.strip().rstrip("/")
    if not base_url:
        raise FrontendError("Set TOMATO_API_URL to the prediction API base URL.")

    try:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, image_bytes, content_type or "application/octet-stream")},
            timeout=HTTP_TIMEOUT,
        )
    except requests.exceptions.ConnectTimeout as exc:
        raise FrontendError("Connection to the prediction service timed out.") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise FrontendError("The prediction service took too long to respond.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise FrontendError("The prediction service is unavailable. Check that FastAPI is running.") from exc
    except requests.exceptions.RequestException as exc:
        raise FrontendError("Could not contact the prediction service. Check TOMATO_API_URL.") from exc

    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError as exc:
            raise FrontendError("The prediction service returned an invalid response.") from exc
        return validate_prediction_response(payload)

    if response.status_code in (400, 422):
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = detail if isinstance(detail, str) else "The image was rejected. Choose a valid image."
        raise FrontendError(message)
    if response.status_code == 413:
        raise FrontendError("The image exceeds the prediction service's upload limit.")
    if response.status_code == 503:
        raise FrontendError("The prediction service is unavailable or its model is not ready.")
    if response.status_code == 500:
        raise FrontendError("The prediction service could not complete this prediction.")
    raise FrontendError(f"The prediction service returned HTTP {response.status_code}.")


def main() -> None:
    st.set_page_config(page_title="Tomato Leaf Classifier", page_icon="🍅")
    st.title("Tomato Leaf Classifier")
    st.warning(
        "Educational classifier: trained on PlantVillage images. It may not generalize "
        "to real-world photographs and is not a validated diagnostic tool."
    )
    st.markdown(
        "**For a clearer image:** use one well-lit tomato leaf against a plain background. "
        "Avoid overlapping leaves, complex backgrounds, and heavy shadows."
    )

    uploaded = st.file_uploader(
        "Upload one tomato leaf image",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        max_upload_size=10,
    )
    if uploaded is None:
        return

    image_bytes = uploaded.getvalue()
    try:
        preview = prepare_preview(image_bytes)
    except FrontendError as exc:
        st.error(str(exc))
        return
    st.image(preview, caption=uploaded.name, width=350)

    if st.button("Predict", type="primary"):
        with st.spinner("Classifying image..."):
            try:
                prediction = request_prediction(
                    image_bytes,
                    uploaded.name,
                    uploaded.type,
                    os.getenv("TOMATO_API_URL", DEFAULT_API_URL),
                )
            except FrontendError as exc:
                st.error(str(exc))
                return

        label = prediction["predicted_class"].removeprefix("Tomato___").replace("_", " ")
        st.success(f"Predicted class: {label}")
        st.table(
            [
                {"Class": name.removeprefix("Tomato___").replace("_", " "), "Probability": f"{score:.1%}"}
                for name, score in prediction["probabilities"].items()
            ]
        )
        st.caption(f"Model version: {prediction['model_version']}")


if __name__ == "__main__":
    main()
