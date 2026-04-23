import io
import json
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from api.auth import (
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET_KEY,
    authenticate,
    current_user,
    home_path_for_role,
    require_role,
)
from api.monitoring import (
    append_prediction_log,
    compute_blur_score,
    compute_brightness,
    get_review_image,
    list_pending_reviews,
    pil_to_rgb_array,
    submit_review,
    utc_now_iso,
)

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_POINTER_PATH = Path("airflow_pipeline/mlruns/current_model.json")

# Fallback artifacts used before the first auto-promotion writes a pointer.
FALLBACK_PCA_PATH = Path("airflow_pipeline/mlruns/1/models/m-99ea752fc1b7401e997dfe755d74e8d6/artifacts/model.pkl")
FALLBACK_CLF_PATH = Path("airflow_pipeline/mlruns/1/models/m-1b25cd047c0d48f3abda57a99fdf9f38/artifacts/model.pkl")


def load_model_pointer() -> tuple[Path, Path, dict]:
    if MODEL_POINTER_PATH.exists():
        with MODEL_POINTER_PATH.open("r", encoding="utf-8") as f:
            pointer = json.load(f)
        return Path(pointer["pca_pkl_path"]), Path(pointer["clf_pkl_path"]), pointer
    return FALLBACK_PCA_PATH, FALLBACK_CLF_PATH, {"source": "fallback"}

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


current_model_info: dict = {}


def load_models():
    global backbone, pca, clf, device, current_model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pca_path, clf_path, pointer = load_model_pointer()
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    current_model_info = {**pointer, "pca_path": str(pca_path), "clf_path": str(clf_path)}
    logging.info("Loaded models from %s / %s", pca_path, clf_path)

    if backbone is None:
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        backbone = backbone.to(device)
        backbone.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    print(f"Models loaded on {device}")
    yield

    global backbone, pca, clf
    del backbone, pca, clf


app = FastAPI(
    title="Places365 Classification API",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=False,
)


STATIC_DIR = Path(__file__).parent / "static"


# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device), "model": current_model_info}


# ── Admin: reload models (reviewer only) ────────────────────────────────────
@app.post("/api/v1/admin/reload")
async def admin_reload(_user: dict = Depends(require_role("reviewer"))):
    try:
        load_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
    return {"status": "success", "model": current_model_info}


# ── Auth endpoints ──────────────────────────────────────────────────────────
@app.post("/api/v1/auth/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = authenticate(username, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["user"] = {"username": user["username"], "role": user["role"]}
    return {"status": "success", "redirect_url": home_path_for_role(user["role"])}


@app.post("/api/v1/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return Response(status_code=204)


@app.get("/api/v1/auth/me")
async def auth_me(request: Request):
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"username": user["username"], "role": user["role"]}


# ── Prediction (user role only) ─────────────────────────────────────────────
@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    _user: dict = Depends(require_role("user")),
):
    start_time = time.perf_counter()

    filename = file.filename.lower() if file.filename else ""
    if filename.endswith(".png"):
        image_mime = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_mime = "image/jpeg"
    else:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Unsupported file type. Use .jpg or .png"},
        )

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid or corrupted image file"},
        )

    width, height = image.size
    rgb = pil_to_rgb_array(image)
    brightness = compute_brightness(rgb)
    blur_score = compute_blur_score(rgb)

    img_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        features = backbone(img_tensor.to(device)).cpu().numpy()

    features_pca = pca.transform(features)
    pred = int(clf.predict(features_pca)[0])
    proba = clf.predict_proba(features_pca)[0]
    confidence = float(proba[pred])

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    predicted_class = CLASS_NAMES[pred]
    requires_manual_review = confidence < CONFIDENCE_THRESHOLD

    append_prediction_log({
        "request_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "filename": file.filename or "",
        "predicted_class": predicted_class,
        "confidence_score": round(confidence, 6),
        "requires_manual_review": requires_manual_review,
        "inference_time_ms": round(elapsed_ms, 2),
        "brightness": round(brightness, 4),
        "blur_score": round(blur_score, 4),
        "width": width,
        "height": height,
        "image_bytes": raw_bytes,
        "image_mime": image_mime,
    })

    return {
        "status": "success",
        "data": {
            "predicted_industry": predicted_class,
            "confidence_score": round(confidence, 4),
            "requires_manual_review": requires_manual_review,
            "inference_time_ms": round(elapsed_ms, 2),
        },
    }


# ── Review endpoints (reviewer role only) ───────────────────────────────────
@app.get("/api/v1/review/pending")
async def review_pending(_user: dict = Depends(require_role("reviewer"))):
    return {"items": list_pending_reviews()}


@app.get("/api/v1/review/{request_id}/image")
async def review_image(
    request_id: str,
    _user: dict = Depends(require_role("reviewer")),
):
    result = get_review_image(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Not found")
    data, mime = result
    return Response(content=data, media_type=mime)


@app.post("/api/v1/review/{request_id}")
async def review_submit(
    request_id: str,
    decision: str = Form(...),
    final_class: str = Form(...),
    _user: dict = Depends(require_role("reviewer")),
):
    if decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")
    if decision == "rejected" and final_class not in CLASS_NAMES.values():
        raise HTTPException(status_code=400, detail="final_class must be one of the known class names")
    ok = submit_review(request_id, decision, final_class)
    if not ok:
        raise HTTPException(status_code=404, detail="No pending review found for that request_id")
    return {"status": "success"}


# ── Gated page routes (must be declared BEFORE static mount) ────────────────
def _serve_static(filename: str) -> FileResponse:
    return FileResponse(STATIC_DIR / filename)


@app.get("/")
async def root(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse(url="/login.html", status_code=303)
    return RedirectResponse(url=home_path_for_role(user["role"]), status_code=303)


@app.get("/login.html")
async def login_page(request: Request):
    user = current_user(request)
    if user is not None:
        return RedirectResponse(url=home_path_for_role(user["role"]), status_code=303)
    return _serve_static("login.html")


@app.get("/index.html")
async def index_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse(url="/login.html", status_code=303)
    if user["role"] != "user":
        return RedirectResponse(url=home_path_for_role(user["role"]), status_code=303)
    return _serve_static("index.html")


@app.get("/review.html")
async def review_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse(url="/login.html", status_code=303)
    if user["role"] != "reviewer":
        return RedirectResponse(url=home_path_for_role(user["role"]), status_code=303)
    return _serve_static("review.html")


# Static assets (non-HTML files served directly). Keep this LAST so the
# explicit routes above take precedence over the fallback mount.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=False), name="static")
