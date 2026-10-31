# Plant Disease Classification — Tomato Leaves (CNN)

A from-scratch convolutional neural network that classifies tomato leaf images
into three categories — **Healthy**, **Early Blight**, and **Late Blight** —
and investigates whether data augmentation improves generalization.

## Overview

| | |
|---|---|
| **Task** | 3-class image classification |
| **Model** | From-scratch CNN (no transfer learning) |
| **Dataset** | PlantVillage (Tomato subset), ~4,500 images |
| **Framework** | TensorFlow / Keras |
| **Research question** | Does data augmentation improve generalization? Which disease classes are hardest to distinguish? |



## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline CNN | 92.70% | 0.929 | 0.927 | 0.928 |
| CNN + Augmentation | 92.04% | 0.922 | 0.920 | 0.920 |

(Weighted averages across the 3 classes. Full output in `outputs/results/model_comparison.csv`.)

**Key finding:** augmentation did not improve accuracy in this run — and more
specifically, it introduced 18 disease-to-Healthy misclassifications (4 Early
Blight, 14 Late Blight) that don't exist in the baseline, which had zero errors
involving the Healthy class. The augmented model's validation curve is also
visibly noisier than baseline's smooth convergence (see
`outputs/figures/training_comparison.png`), suggesting augmentation added
training instability without a corresponding generalization benefit — plausibly
because baseline wasn't significantly overfitting to begin with, so
augmentation's usual benefit (reducing overfitting) had less to correct.

## Dataset

- **Source:** [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé, 2015–2016), `raw/color/` subset
- **Classes:** `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`
- **Counts:** Healthy ~1,591 / Early Blight ~1,000 / Late Blight ~1,909 (~4,500 total)
- **Split:** 80% train / 10% validation / 10% test, fixed seed (42)

## Model Architecture

```
Input (128x128x3)
     |
Conv2D(32) -> ReLU -> MaxPooling
     |
Conv2D(64) -> ReLU -> MaxPooling
     |
Flatten -> Dense(128) -> ReLU -> Dropout(0.5) -> Dense(3) -> Softmax
```

Trained with Adam, categorical cross-entropy, early stopping (patience=3 on
validation loss, max 20 epochs).

## Project Structure

```
.
├── src/
│   ├── data_prep.py   # verify dataset, EDA, train/val/test split
│   ├── model.py        # CNN architecture (shared, not run directly)
│   ├── train.py         # trains baseline + augmented models
│   └── evaluate.py      # evaluates both models, builds comparison table
├── data/
│   ├── raw/              # downloaded dataset goes here
│   └── split/            # created by data_prep.py
├── outputs/
│   ├── figures/           # all plots
│   ├── models/             # saved .keras models
│   └── results/             # CSVs: training history, comparison table
├── README.md
├── requirements.txt

```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get the dataset (no Kaggle account needed — this only downloads the
   ~4,500 tomato images, not the full multi-GB repo):
   ```
   git clone --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset.git
   cd PlantVillage-Dataset
   git sparse-checkout init --cone
   git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight
   git checkout master
   cd ..
   mkdir -p data/raw
   cp -r PlantVillage-Dataset/raw/color/Tomato___healthy data/raw/
   cp -r PlantVillage-Dataset/raw/color/Tomato___Early_blight data/raw/
   cp -r PlantVillage-Dataset/raw/color/Tomato___Late_blight data/raw/
   rm -rf PlantVillage-Dataset
   ```

## Run

From the project root (not from inside `src/`):

```
python src/data_prep.py
python src/train.py
python src/evaluate.py
```

## Predict one image with the saved baseline model

From the project root, run:

```
python -m src.predict_local "path/to/tomato-leaf.jpg"
```

The command loads `outputs/models/model_baseline.keras` without training or
running evaluation, then prints JSON with `predicted_class` and a probability
for each class. Accepted image formats are JPEG (`.jpg` or `.jpeg`), PNG, and
BMP. The output index order is `Tomato___Early_blight`,
`Tomato___Late_blight`, then `Tomato___healthy`. Images are converted to RGB,
resized to 128×128 with nearest-neighbor interpolation, and scaled by `1/255`,
matching evaluation. Invalid or unreadable images produce an error and a
nonzero exit code.

## Inference API

Install the API dependencies and start one local server process from the
project root:

```
pip install -r requirements-inference.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for the API documentation. Example requests
from a terminal with `curl` (use `curl.exe` in PowerShell):

```
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/version
curl -F "file=@path/to/tomato-leaf.jpg" http://127.0.0.1:8000/predict
```

`POST /predict` accepts one JPEG, PNG, or BMP image in the multipart `file`
field, up to 10 MiB and 20 million decoded pixels. It returns the predicted
class, probabilities for all three classes, and a short hash-based
`model_version`. `GET /version` returns that identifier, the full model
artifact SHA-256, and `application_version` (`0.1.0`). `GET /health` reports
process liveness; `GET /ready` returns 503 if the baseline model did not load
at startup. Invalid images return 400, oversized images 413, and unexpected
inference failures 500. Restart the server after correcting a model-loading
failure.

## Tests

Install the inference, frontend, and test dependencies, then run the fast suite
from the project root. These tests use generated images and mocks; they do not
need the dataset or a model artifact:

```
pip install -r requirements-inference.txt -r requirements-frontend.txt -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

Run the saved-model smoke test separately when
`outputs/models/model_baseline.keras` is available. It loads the real model
and predicts one generated image, without checking a disease label:

```
python -m unittest discover -s tests -p "model_smoke.py" -v
```

The fast suite is the intended required CI check; the model smoke test can run
as a separate artifact-aware check. Backend tests import TensorFlow, while the
frontend test file can run with frontend dependencies alone.

## Streamlit frontend

The frontend calls the FastAPI service over HTTP and does not load the model.
Install its separate dependencies:

```
pip install -r requirements-frontend.txt
```

Start the API as shown above in one terminal. In a second terminal, run:

```
python -m streamlit run src/frontend.py --server.port 8501
```

Open `http://localhost:8501`, upload one JPG, JPEG, PNG, or BMP image, and
click **Predict**. The frontend previews readable images and shows the class,
all three probabilities, and model version. Uploads are limited to 10 MiB;
images that cannot be previewed safely are rejected before the API call.

The frontend defaults to `http://127.0.0.1:8000` for the API. To use another
backend in PowerShell, set `$env:TOMATO_API_URL="http://host:port"` before
starting Streamlit. The API's Swagger UI remains available at
`http://127.0.0.1:8000/docs` when using the default backend.

## Outputs

- `outputs/figures/class_distribution.png` — image counts per class
- `outputs/figures/example_images.png` — sample images per class
- `outputs/models/model_baseline.keras` / `model_augmented.keras` — trained models
- `outputs/results/baseline_history.csv` / `augmented_history.csv` — training logs
- `outputs/results/model_comparison.csv` — final Accuracy/Precision/Recall/F1 table
- `outputs/figures/training_comparison.png` — accuracy/loss curves, baseline vs. augmented
- `outputs/figures/confusion_matrix.png` — confusion matrix (augmented model)
- `outputs/figures/misclassified_examples.png` — example errors, actual vs. predicted

## Known Limitations

- Train/val/test splits are per-image random splits (fixed seed), not grouped
  by source leaf. PlantVillage is known to contain near-duplicate images of
  the same physical leaf; leaf-grouped splitting was considered and
  deliberately not implemented, to keep the project's scope aligned with its
  from-scratch, beginner-level goals.
- Results reflect a single training run per model (no fixed random seed for
  weight initialization). Given the visible instability in the augmented
  model's validation curve, re-running would likely shift the exact numbers
  somewhat — treat the specific percentages as indicative, not exact.

## Model Limitations and Image Guidelines

This CNN was trained and evaluated on the PlantVillage tomato leaf dataset, which primarily contains individual leaves photographed against relatively uniform backgrounds.

For more reliable predictions, users are encouraged to:
- Upload a clear image of a single tomato leaf.
- Prefer a plain, uniform background similar to PlantVillage images.
- Ensure the leaf is well-lit and clearly visible.
- Avoid images containing multiple overlapping leaves, complex backgrounds, or excessive shadows.

**Important:** The model has not been validated for reliable performance on real-world field photographs. Predictions are experimental and should not be treated as definitive plant-disease diagnoses.

## Acknowledgments

Dataset: PlantVillage (Hughes & Salathé, 2015–2016), sourced via
[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset).
