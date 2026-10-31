"""Focused tests for the Streamlit client's validation and API calls."""

import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image

from src import frontend


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (120, 200, 40)).save(output, format="PNG")
    return output.getvalue()


def valid_response() -> dict[str, object]:
    return {
        "predicted_class": "Tomato___healthy",
        "probabilities": {
            "Tomato___Early_blight": 0.1,
            "Tomato___Late_blight": 0.2,
            "Tomato___healthy": 0.7,
        },
        "model_version": "baseline-abc123",
    }


class FrontendTests(unittest.TestCase):
    def test_preview_checks_size_readability_and_dimensions(self):
        image = png_bytes()
        preview = frontend.prepare_preview(image)
        self.assertEqual(preview.size, (4, 3))
        with self.assertRaisesRegex(frontend.FrontendError, "empty"):
            frontend.prepare_preview(b"")
        with patch.object(frontend, "MAX_UPLOAD_BYTES", len(image) - 1):
            with self.assertRaisesRegex(frontend.FrontendError, "10 MiB"):
                frontend.prepare_preview(image)
        with patch.object(frontend, "MAX_PREVIEW_PIXELS", 11):
            with self.assertRaisesRegex(frontend.FrontendError, "too large"):
                frontend.prepare_preview(image)
        with self.assertRaisesRegex(frontend.FrontendError, "unreadable"):
            frontend.prepare_preview(b"broken image")

    @patch("src.frontend.requests.post")
    def test_http_upload_and_validated_response(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = valid_response()
        image = png_bytes()

        result = frontend.request_prediction(image, "leaf.png", "image/png", "http://localhost:8000/")

        self.assertEqual(result, valid_response())
        post.assert_called_once_with(
            "http://localhost:8000/predict",
            files={"file": ("leaf.png", image, "image/png")},
            timeout=(3, 30),
        )

    @patch("src.frontend.requests.post")
    def test_oversized_image_never_reaches_backend(self, post):
        with patch.object(frontend, "MAX_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(frontend.FrontendError, "10 MiB"):
                frontend.request_prediction(b"1234", "leaf.png", "image/png", frontend.DEFAULT_API_URL)
        post.assert_not_called()

    @patch("src.frontend.requests.post")
    def test_backend_error_responses(self, post):
        cases = {
            400: "bad image",
            413: "upload limit",
            422: "rejected",
            503: "unavailable",
            500: "could not complete",
            502: "HTTP 502",
        }
        for status, message in cases.items():
            with self.subTest(status=status):
                post.return_value.status_code = status
                post.return_value.json.return_value = {"detail": "bad image" if status == 400 else []}
                with self.assertRaises(frontend.FrontendError) as caught:
                    frontend.request_prediction(b"image", "leaf.png", "image/png", frontend.DEFAULT_API_URL)
                self.assertIn(message, str(caught.exception))

    @patch("src.frontend.requests.post")
    def test_connection_and_timeout_errors(self, post):
        cases = (
            (requests.exceptions.ConnectionError("offline"), "unavailable"),
            (requests.exceptions.ConnectTimeout("slow connect"), "Connection"),
            (requests.exceptions.ReadTimeout("slow response"), "too long"),
            (requests.exceptions.InvalidURL("bad URL"), "TOMATO_API_URL"),
        )
        for error, message in cases:
            with self.subTest(error=type(error).__name__):
                post.side_effect = error
                with self.assertRaises(frontend.FrontendError) as caught:
                    frontend.request_prediction(b"image", "leaf.png", "image/png", frontend.DEFAULT_API_URL)
                self.assertIn(message, str(caught.exception))

    def test_rejects_malformed_success_responses(self):
        invalid = [
            None,
            {"probabilities": valid_response()["probabilities"], "model_version": "version"},
            {**valid_response(), "model_version": ""},
            {**valid_response(), "probabilities": {"a": 0.5, "b": 0.5}},
            {**valid_response(), "probabilities": {"a": 0.1, "b": 0.2, "c": 0.7}},
            {**valid_response(), "probabilities": {"a": True, "b": 0.2, "Tomato___healthy": 0.8}},
            {**valid_response(), "probabilities": {"a": math.inf, "b": 0.2, "Tomato___healthy": 0.8}},
            {**valid_response(), "probabilities": {"a": -0.1, "b": 0.2, "Tomato___healthy": 0.9}},
            {**valid_response(), "probabilities": {"a": 10**400, "b": 0.2, "Tomato___healthy": 0.8}},
            {**valid_response(), "probabilities": {"a": 0.1, "b": 0.2, "Tomato___healthy": 0.2}},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(frontend.FrontendError):
                frontend.validate_prediction_response(payload)

    @patch("src.frontend.requests.post")
    def test_malformed_http_json_is_handled(self, post):
        post.return_value.status_code = 200
        post.return_value.json.side_effect = ValueError("not JSON")
        with self.assertRaisesRegex(frontend.FrontendError, "invalid response"):
            frontend.request_prediction(b"image", "leaf.png", "image/png", frontend.DEFAULT_API_URL)

    @patch("src.frontend.request_prediction", return_value=valid_response())
    @patch("src.frontend.st")
    def test_button_is_the_only_request_trigger(self, streamlit, request_prediction):
        image = png_bytes()
        streamlit.file_uploader.return_value = SimpleNamespace(
            getvalue=lambda: image, name="leaf.png", type="image/png"
        )
        streamlit.button.side_effect = (False, True, False)

        frontend.main()
        request_prediction.assert_not_called()
        frontend.main()
        request_prediction.assert_called_once()
        frontend.main()
        request_prediction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
