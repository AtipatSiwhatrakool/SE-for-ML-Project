from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

# Host-side path. When FastAPI runs from repo root, this points to the shared folder.
DRIFT_DIR = Path("airflow_pipeline/data/drift")
PREDICTION_LOG_CSV = DRIFT_DIR / "prediction_logs.csv"
BASELINE_JSON = DRIFT_DIR / "baseline_reference.json"
REPORTS_DIR = DRIFT_DIR / "reports"


def ensure_monitoring_dirs() -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_brightness(rgb: np.ndarray) -> float:
    # Perceived luminance
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return float(gray.mean())


def compute_blur_score(rgb: np.ndarray) -> float:
    """
    Blur proxy using variance of a Laplacian-like response.
    Higher = sharper, Lower = blurrier.
    Implemented without OpenCV.
    """
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:] -
        4.0 * padded[1:-1, 1:-1]
    )
    return float(lap.var())


def append_prediction_log(row: Dict[str, Any]) -> None:
    ensure_monitoring_dirs()

    fieldnames = [
        "request_id",
        "timestamp",
        "filename",
        "predicted_class",
        "confidence_score",
        "requires_manual_review",
        "inference_time_ms",
        "brightness",
        "blur_score",
        "width",
        "height",
    ]

    file_exists = PREDICTION_LOG_CSV.exists()
    with PREDICTION_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)