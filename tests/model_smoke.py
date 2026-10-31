"""Opt-in smoke test for the saved baseline model artifact."""

import io
import math
import unittest

from PIL import Image

from src.inference import CLASS_NAMES, load_baseline_model, predict_image_bytes


class ModelSmokeTests(unittest.TestCase):
    def test_saved_baseline_model_predicts_generated_image(self):
        image = io.BytesIO()
        Image.new("RGB", (128, 128), (80, 150, 60)).save(image, format="PNG")

        model = load_baseline_model()
        result = predict_image_bytes(image.getvalue(), model)

        self.assertEqual(set(result["probabilities"]), set(CLASS_NAMES))
        self.assertIn(result["predicted_class"], CLASS_NAMES)
        scores = list(result["probabilities"].values())
        self.assertTrue(all(math.isfinite(score) and 0 <= score <= 1 for score in scores))
        self.assertAlmostEqual(sum(scores), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
