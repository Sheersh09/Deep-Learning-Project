"""Focused API tests without a model artifact or dataset download."""

import asyncio
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from src import api, inference


def make_request(application: FastAPI) -> Request:
    return Request({"type": "http", "app": application, "method": "GET", "path": "/", "headers": []})


def make_upload(contents: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(contents), filename="leaf.png")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.application = FastAPI()
        self.application.state.model = object()
        self.application.state.model_version = "baseline-abc123"
        self.application.state.artifact_sha256 = "abc123" * 10 + "abcd"
        self.request = make_request(self.application)

    def test_health_is_independent_of_model(self):
        self.assertEqual(api.health(), {"status": "ok"})
        self.application.state.model = None
        self.assertEqual(api.health(), {"status": "ok"})

    def test_ready_and_version(self):
        self.assertEqual(api.ready(self.request), {"status": "ready"})
        self.assertEqual(
            api.version(self.request).model_dump(),
            {
                "application_version": "0.1.0",
                "model_version": "baseline-abc123",
                "artifact_sha256": "abc123" * 10 + "abcd",
            },
        )
        self.application.state.model = None
        for endpoint in (api.ready, api.version):
            with self.subTest(endpoint=endpoint.__name__), self.assertRaises(HTTPException) as caught:
                endpoint(self.request)
            self.assertEqual(caught.exception.status_code, 503)

    def test_lifespan_loads_once_and_hashes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.keras"
            artifact.write_bytes(b"saved baseline artifact")
            expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            model = object()

            async def check_lifespan():
                with patch.object(api, "DEFAULT_MODEL_PATH", artifact), patch.object(
                    api, "load_baseline_model", return_value=model
                ) as loader:
                    async with api.lifespan(self.application):
                        loader.assert_called_once_with()
                        self.assertIs(self.application.state.model, model)
                        self.assertEqual(self.application.state.artifact_sha256, expected_hash)
                        self.assertEqual(
                            self.application.state.model_version,
                            f"baseline-{expected_hash[:12]}",
                        )
                    self.assertIsNone(self.application.state.model)

            asyncio.run(check_lifespan())

    def test_lifespan_load_failure_keeps_health_but_blocks_other_routes(self):
        async def check_lifespan():
            with patch.object(api, "load_baseline_model", side_effect=FileNotFoundError("missing")):
                with self.assertLogs(api.logger, level="ERROR"):
                    async with api.lifespan(self.application):
                        self.assertEqual(api.health(), {"status": "ok"})
                        for endpoint in (api.ready, api.version):
                            with self.assertRaises(HTTPException) as caught:
                                endpoint(self.request)
                            self.assertEqual(caught.exception.status_code, 503)
                        with self.assertRaises(HTTPException) as caught:
                            api.predict(self.request, make_upload(b"image"))
                        self.assertEqual(caught.exception.status_code, 503)

        asyncio.run(check_lifespan())

    def test_valid_prediction_and_documented_warning(self):
        result = {
            "predicted_class": inference.CLASS_NAMES[1],
            "probabilities": {name: 1.0 if i == 1 else 0.0 for i, name in enumerate(inference.CLASS_NAMES)},
        }
        with patch.object(api, "predict_image_bytes", return_value=result) as predictor:
            response = api.predict(self.request, make_upload(b"image bytes"))
        predictor.assert_called_once_with(b"image bytes", self.application.state.model)
        self.assertEqual(response.predicted_class, inference.CLASS_NAMES[1])
        self.assertEqual(response.probabilities, result["probabilities"])
        self.assertEqual(response.model_version, "baseline-abc123")
        self.assertIn("educational classifier", api.app.openapi()["paths"]["/predict"]["post"]["description"])

    def test_oversized_upload(self):
        with patch.object(api, "MAX_UPLOAD_BYTES", 3), patch.object(api, "predict_image_bytes") as predictor:
            with self.assertRaises(HTTPException) as caught:
                api.predict(self.request, make_upload(b"1234"))
        self.assertEqual(caught.exception.status_code, 413)
        predictor.assert_not_called()

    def test_empty_corrupt_and_unsupported_images(self):
        unsupported = io.BytesIO()
        Image.new("RGB", (2, 2)).save(unsupported, format="GIF")
        for contents in (b"", b"not an image", unsupported.getvalue()):
            with self.subTest(contents=contents[:12]), self.assertRaises(HTTPException) as caught:
                api.predict(self.request, make_upload(contents))
            self.assertEqual(caught.exception.status_code, 400)

    def test_unexpected_prediction_error_is_logged_and_hidden(self):
        with patch.object(api, "predict_image_bytes", side_effect=RuntimeError("private detail")):
            with self.assertLogs(api.logger, level="ERROR") as logs:
                with self.assertRaises(HTTPException) as caught:
                    api.predict(self.request, make_upload(b"image"))
        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(caught.exception.detail, "Prediction failed.")
        self.assertNotIn("private detail", caught.exception.detail)
        self.assertIn("Unexpected inference failure", " ".join(logs.output))

    def test_http_routes_with_loaded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.keras"
            artifact.write_bytes(b"test artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            model = object()
            prediction = {
                "predicted_class": inference.CLASS_NAMES[0],
                "probabilities": {name: float(index == 0) for index, name in enumerate(inference.CLASS_NAMES)},
            }
            with patch.object(api, "DEFAULT_MODEL_PATH", artifact), patch.object(
                api, "load_baseline_model", return_value=model
            ) as loader, patch.object(api, "predict_image_bytes", return_value=prediction) as predictor:
                with TestClient(api.app) as client:
                    loader.assert_called_once_with()
                    self.assertEqual(client.get("/health").json(), {"status": "ok"})
                    self.assertEqual(client.get("/ready").json(), {"status": "ready"})
                    version = client.get("/version")
                    self.assertEqual(version.status_code, 200)
                    self.assertEqual(version.json(), {
                        "application_version": "0.1.0",
                        "model_version": f"baseline-{digest[:12]}",
                        "artifact_sha256": digest,
                    })

                    response = client.post("/predict", files={"file": ("leaf.png", b"image", "image/png")})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**prediction, "model_version": f"baseline-{digest[:12]}"})
                    predictor.assert_called_once_with(b"image", model)

                    self.assertEqual(client.post("/predict").status_code, 422)
                    predictor.side_effect = inference.InvalidImageError("Image is unreadable or corrupt.")
                    invalid = client.post("/predict", files={"file": ("leaf.png", b"bad", "image/png")})
                    self.assertEqual(invalid.status_code, 400)
                    self.assertEqual(invalid.json(), {"detail": "Image is unreadable or corrupt."})
                    with patch.object(api, "MAX_UPLOAD_BYTES", 3):
                        oversized = client.post("/predict", files={"file": ("leaf.png", b"1234", "image/png")})
                    self.assertEqual(oversized.status_code, 413)
                    self.assertEqual(predictor.call_count, 2)

    def test_http_routes_when_model_load_fails(self):
        with patch.object(api, "load_baseline_model", side_effect=FileNotFoundError("missing")) as loader:
            with self.assertLogs(api.logger, level="ERROR"):
                with TestClient(api.app) as client:
                    loader.assert_called_once_with()
                    self.assertEqual(client.get("/health").status_code, 200)
                    for path in ("/ready", "/version"):
                        with self.subTest(path=path):
                            response = client.get(path)
                            self.assertEqual(response.status_code, 503)
                            self.assertEqual(response.json(), {"detail": "Model unavailable."})
                    response = client.post("/predict", files={"file": ("leaf.png", b"image", "image/png")})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json(), {"detail": "Model unavailable."})


if __name__ == "__main__":
    unittest.main()
