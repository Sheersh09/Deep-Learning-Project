# Tomato Leaf Classifier — ML Deployment Project

A trained CNN is only useful if people can run it. I’m using this project to take an existing TensorFlow model through the practical steps of deployment: a local inference service, a browser client, clear failure handling, tests, and eventually reproducible releases.

The app classifies a tomato leaf as **early blight**, **late blight**, or **healthy**. It is an educational prototype, not a validated disease diagnostic tool.

## What runs today

- **FastAPI** loads the saved baseline model at startup and serves predictions. `/health`, `/ready`, and `/version` expose liveness, model readiness, and artifact identity.
- **Streamlit** previews an image, sends it to the API over HTTP, and displays the class, three probabilities, and model version. The UI does not load TensorFlow.
- **TensorFlow/Keras** provides the from-scratch CNN. Local CLI prediction and the API share the same inference preprocessing.
- **Tests** cover inference, API behavior, and frontend validation; a separate smoke test loads the real model artifact.

```text
Leaf image → Streamlit → FastAPI → saved CNN → class + probabilities
```

This application also runs locally with Docker Compose. Both service images and containers have been built and verified. CI/CD, public cloud deployment, and monitoring are planned work.

## Try it locally

From the repository root, create an environment and install the app dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-inference.txt -r requirements-frontend.txt
```

Run the API and UI in separate terminals (activate the environment in both):

```powershell
# Terminal 1
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1

# Terminal 2
python -m streamlit run src/frontend.py --server.port 8501
```

Open `http://localhost:8501` and upload one JPG/JPEG, PNG, or BMP image. API docs are at `http://127.0.0.1:8000/docs`. The saved model is already under `outputs/models/`; no training dataset is needed to run the app. For CLI prediction, use `python -m src.predict_local "path/to/leaf.jpg"`.

On macOS/Linux, replace the first two setup commands with `python3 -m venv .venv` and `source .venv/bin/activate`.

## Run with Docker Compose

From the repository root, build and start the two services:

```powershell
docker compose config
docker compose build
docker compose up -d --wait
docker compose ps
```

Open `http://localhost:8501` for the frontend and `http://localhost:8000/docs` for the API. Upload one tomato leaf image and click **Predict**. The backend image includes the tracked baseline model; neither image needs the training dataset. Compose sets `TOMATO_API_URL=http://backend:8000` so the frontend reaches the API over the Compose network. The backend health check uses `/ready`, which stays unavailable if the model fails to load, and the frontend starts only after the backend is healthy.

To inspect logs or stop the services:

```powershell
docker compose logs --tail=100 backend frontend
docker compose down
```

Both services run as non-root users. Ports 8000 and 8501 are published only on the host's loopback interface for local development. The separate Python workflow above remains available.

## Engineering details

The service accepts one image per request, with a 10 MiB upload limit and a 20-million-pixel decoded-image limit. It converts valid images to 128 × 128 RGB and scales pixels by `1/255`, matching evaluation. `/ready` returns 503 if the model did not load; `/version` reports a SHA-256-based model identifier. The frontend reports upload and service errors to the user.

The application code lives in `src/api.py`, `src/inference.py`, and `src/frontend.py`. Training and evaluation scripts are also in `src/`, but are separate from the serving path. Saved models and recorded metrics live under `outputs/`.

## Model scope

The saved baseline CNN recorded **92.70% accuracy** and **0.928 weighted F1** on one held-out PlantVillage tomato-leaf test split ([recorded metrics](outputs/results/model_comparison.csv)). These numbers describe that evaluation only. The 80/10/10 split is image-level rather than leaf-grouped, so similar photos may cross splits; training was run once without a fixed weight-initialization seed. PlantVillage’s plain backgrounds also differ from field photographs. **Do not treat a prediction as a definitive plant-disease diagnosis.**

For best results with this experimental model, use one clear, well-lit tomato leaf against a plain background. The dataset comes from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes and Salathé).

## Checks

The fast suite uses generated images and mocks, so it needs no dataset or model artifact. The separate smoke test loads the saved baseline model and checks that inference completes.

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s tests -p "model_smoke.py" -v
```

## Roadmap

- [x] Local inference API, browser client, and automated tests
- [x] Verify the Docker Compose build and runtime for the API and UI
- [ ] Add CI checks and a release pipeline
- [ ] Deploy to cloud infrastructure and add monitoring
- [ ] Explore service boundaries as the deployment grows
