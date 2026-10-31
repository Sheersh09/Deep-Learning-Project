"""Focused tests for saved-model inference without training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from src import inference

from src.inference import (
    CLASS_NAMES,
    load_baseline_model,
    predict_image,
    predict_image_bytes,
)
from src.predict_local import main


class StubModel:
    def __init__(self, scores=(0.1, 0.8, 0.1)):
        self.scores = np.array([scores], dtype=np.float32)
        self.batch = None

    def predict(self, batch, verbose=0):
        self.batch = batch
        return self.scores


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.image_path = Path(self.temp_dir.name) / "leaf.png"
        Image.new("RGB", (8, 4), (255, 128, 0)).save(self.image_path)

    def test_prediction_mapping_and_preprocessing(self):
        model = StubModel()
        result = predict_image(self.image_path, model)

        self.assertEqual(result["predicted_class"], CLASS_NAMES[1])
        self.assertEqual(list(result["probabilities"]), list(CLASS_NAMES))
        self.assertAlmostEqual(result["probabilities"][CLASS_NAMES[1]], 0.8)
        self.assertEqual(model.batch.shape, (1, 128, 128, 3))
        self.assertEqual(model.batch.dtype, np.float32)
        np.testing.assert_allclose(model.batch[0, 0, 0], [1.0, 128 / 255, 0.0])

    def test_bytes_and_path_use_the_same_preprocessing(self):
        path_model = StubModel()
        bytes_model = StubModel()

        predict_image(self.image_path, path_model)
        predict_image_bytes(self.image_path.read_bytes(), bytes_model)

        np.testing.assert_array_equal(path_model.batch, bytes_model.batch)

    def test_grayscale_image_becomes_three_channels(self):
        path = Path(self.temp_dir.name) / "gray.png"
        Image.new("L", (3, 3), 64).save(path)
        model = StubModel()

        predict_image(path, model)

        self.assertEqual(model.batch.shape, (1, 128, 128, 3))
        np.testing.assert_allclose(model.batch[0, 0, 0], [64 / 255] * 3)

    def test_invalid_paths_and_image_formats(self):
        model = StubModel()
        with self.assertRaises(FileNotFoundError):
            predict_image(Path(self.temp_dir.name) / "missing.jpg", model)
        with self.assertRaisesRegex(ValueError, "not a file"):
            predict_image(Path(self.temp_dir.name), model)

        unsupported = Path(self.temp_dir.name) / "leaf.gif"
        Image.new("RGB", (2, 2)).save(unsupported)
        with self.assertRaisesRegex(ValueError, "extension"):
            predict_image(unsupported, model)
        with self.assertRaisesRegex(ValueError, "format"):
            predict_image_bytes(unsupported.read_bytes(), model)

        mismatched = Path(self.temp_dir.name) / "leaf.jpg"
        mismatched.write_bytes(self.image_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "does not match"):
            predict_image(mismatched, model)

    def test_unreadable_and_empty_inputs(self):
        model = StubModel()
        corrupt = Path(self.temp_dir.name) / "broken.png"
        corrupt.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "unreadable or corrupt"):
            predict_image(corrupt, model)
        with self.assertRaisesRegex(ValueError, "empty"):
            predict_image_bytes(b"", model)

    def test_decoded_pixel_limit(self):
        with patch.object(inference, "MAX_IMAGE_PIXELS", 3):
            with self.assertRaisesRegex(inference.InvalidImageError, "decoded-pixel limit"):
                predict_image_bytes(self.image_path.read_bytes(), StubModel())

    def test_rejects_wrong_model_output_shape(self):
        with self.assertRaisesRegex(ValueError, "invalid prediction"):
            predict_image(self.image_path, StubModel((0.5, 0.5)))

    def test_missing_model_artifact(self):
        with self.assertRaises(FileNotFoundError):
            load_baseline_model(Path(self.temp_dir.name) / "missing.keras")

    @patch("src.predict_local.predict_image")
    @patch("src.predict_local.load_baseline_model")
    def test_cli_prints_json(self, load_mock, predict_mock):
        predict_mock.return_value = {
            "predicted_class": CLASS_NAMES[0],
            "probabilities": {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(CLASS_NAMES)},
        }
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([str(self.image_path)])

        self.assertEqual(exit_code, 0)
        load_mock.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), predict_mock.return_value)

    @patch("src.predict_local.load_baseline_model", side_effect=FileNotFoundError("model missing"))
    def test_cli_reports_errors(self, _load_mock):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            exit_code = main([str(self.image_path)])
        self.assertEqual(exit_code, 2)
        self.assertIn("model missing", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
