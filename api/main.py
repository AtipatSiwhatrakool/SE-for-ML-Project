import os
import time
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ── Paths ───────────────────────────────────────────────────────────────────
PCA_PATH = Path("airflow_pipeline/mlruns/1/models/m-99ea752fc1b7401e997dfe755d74e8d6/artifacts/model.pkl")
CLF_PATH = Path("airflow_pipeline/mlruns/1/models/m-1b25cd047c0d48f3abda57a99fdf9f38/artifacts/model.pkl")

# ── Class Mapping ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "beauty_salon",
    1: "drugstore",
    2: "restaurant",
    3: "movie_theater",
    4: "apartment_building",
    5: "supermarket",
}
CONFIDENCE_THRESHOLD = 0.60

# ── Global model references (loaded once at startup) ────────────────────────
backbone = None
pca = None
clf = None
device = None

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_models():
    global backbone, pca, clf, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PCA
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)

    # Load classifier
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    # Load EfficientNetV2-S backbone
    backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    load_models()
    print(f"Models loaded on {device}")
    yield
    # Shutdown: cleanup
    global backbone, pca, clf
    del backbone, pca, clf


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Places365 Classification API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}


@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    # Validate file type by extension
    filename = file.filename.lower() if file.filename else ""
    if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Unsupported file type. Use .jpg or .png"}
        )

    # Load and preprocess image
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # Extract features
    with torch.inference_mode():
        features = backbone(img_tensor.to(device)).cpu().numpy()

    # Apply PCA
    features_pca = pca.transform(features)

    # Predict
    pred = clf.predict(features_pca)[0]
    proba = clf.predict_proba(features_pca)[0]
    confidence = float(proba[pred])

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "success",
        "data": {
            "predicted_industry": CLASS_NAMES[pred],
            "confidence_score": round(confidence, 4),
            "requires_manual_review": confidence < CONFIDENCE_THRESHOLD,
            "inference_time_ms": round(elapsed_ms, 2),
        }
    }


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
