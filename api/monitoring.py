from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
                brightness, blur_score, width, height,
                image_bytes, image_mime
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                psycopg2.Binary(row["image_bytes"]),
                row["image_mime"],
            ),
        )


def list_pending_reviews() -> list[Dict[str, Any]]:
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT request_id, timestamp, filename, predicted_class,
                   confidence_score, inference_time_ms,
                   brightness, blur_score, width, height, image_mime
            FROM prediction_logs
            WHERE review_status = 'pending'
            ORDER BY timestamp ASC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "request_id": str(r[0]),
            "timestamp": r[1].isoformat() if r[1] else None,
            "filename": r[2],
            "predicted_class": r[3],
            "confidence_score": float(r[4]),
            "inference_time_ms": float(r[5]),
            "brightness": float(r[6]),
            "blur_score": float(r[7]),
            "width": int(r[8]),
            "height": int(r[9]),
            "image_mime": r[10],
        }
        for r in rows
    ]


def get_review_image(request_id: str) -> Optional[tuple[bytes, str]]:
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT image_bytes, image_mime FROM prediction_logs WHERE request_id = %s",
            (request_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return bytes(row[0]), row[1]


def submit_review(request_id: str, decision: str, final_class: Optional[str]) -> bool:
    if decision not in ("approved", "rejected"):
        raise ValueError(f"Invalid decision: {decision}")
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE prediction_logs
               SET review_status = %s,
                   final_class = %s,
                   reviewed_at = NOW()
             WHERE request_id = %s AND review_status = 'pending'
            """,
            (decision, final_class, request_id),
        )
        updated = cur.rowcount
    return updated > 0


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_latest_drift_report() -> Dict[str, Any]:
    ensure_monitoring_dirs()
    latest_path = REPORTS_DIR / "latest_report.json"
    if not latest_path.exists():
        return {
            "status": "not_available",
            "drift_detected": False,
            "message": "No drift report has been generated yet.",
        }

    with latest_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if "message" not in report:
        status = str(report.get("status", "unknown"))
        if status == "ok":
            report["message"] = "Latest production drift report loaded."
        elif status == "insufficient_recent_rows":
            recent = report.get("recent_rows", 0)
            required = report.get("required_rows", 0)
            report["message"] = (
                f"Need at least {required} recent rows for drift detection; only {recent} available."
            )
        elif status == "empty_prediction_logs":
            report["message"] = "Prediction logs are empty, so drift cannot be evaluated yet."
        else:
            report["message"] = "Drift report loaded."

    return report
