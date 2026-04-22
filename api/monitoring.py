from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psycopg2
from PIL import Image

PREDICTIONS_DATABASE_URL = os.getenv(
    "PREDICTIONS_DATABASE_URL",
    "postgresql://predictions:predictions@localhost:5433/predictions",
)

DRIFT_DIR = Path("airflow_pipeline/data/drift")
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
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_logs (
                request_id, timestamp, filename, predicted_class,
                confidence_score, requires_manual_review, inference_time_ms,
                brightness, blur_score, width, height
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["request_id"],
                row["timestamp"],
                row.get("filename"),
                row["predicted_class"],
                row["confidence_score"],
                row["requires_manual_review"],
                row["inference_time_ms"],
                row["brightness"],
                row["blur_score"],
                row["width"],
                row["height"],
            ),
        )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
