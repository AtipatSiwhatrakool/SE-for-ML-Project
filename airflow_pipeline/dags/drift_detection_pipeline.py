import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psycopg2
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Airflow-side shared path (baseline + reports still land on the bind-mounted volume)
DRIFT_DIR = Path("/opt/airflow/data/drift")
BASELINE_JSON = DRIFT_DIR / "baseline_reference.json"
REPORTS_DIR = DRIFT_DIR / "reports"

PREDICTIONS_DATABASE_URL = os.getenv(
    "PREDICTIONS_DATABASE_URL",
    "postgresql://predictions:predictions@predictions-db:5432/predictions",
)

# Current repo classes (6 classes). Update if your trained model changes.
CLASS_NAMES = [
    "beauty_salon",
    "drugstore",
    "restaurant",
    "movie_theater",
    "apartment_building",
    "supermarket",
]

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.25"))
WINDOW_DAYS = int(os.getenv("DRIFT_WINDOW_DAYS", "7"))
MIN_ROWS_FOR_DRIFT = int(os.getenv("DRIFT_MIN_ROWS", "20"))

# Rejection-rate signal over the same window
REJECTION_RATE_THRESHOLD = float(os.getenv("REJECTION_RATE_THRESHOLD", "0.30"))
MIN_REVIEWED_IN_WINDOW = int(os.getenv("MIN_REVIEWED_IN_WINDOW", "10"))

# Trigger smoothing
RETRAIN_COOLDOWN_DAYS = int(os.getenv("RETRAIN_COOLDOWN_DAYS", "7"))
DRIFT_PERSISTENCE_WINDOW = int(os.getenv("DRIFT_PERSISTENCE_WINDOW", "5"))
DRIFT_PERSISTENCE_K = int(os.getenv("DRIFT_PERSISTENCE_K", "3"))


def _ensure_dirs():
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_logs() -> pd.DataFrame:
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn:
        df = pd.read_sql_query(
            """
            SELECT request_id, timestamp, filename, predicted_class,
                   confidence_score, requires_manual_review, inference_time_ms,
                   brightness, blur_score, width, height
            FROM prediction_logs
            ORDER BY timestamp ASC
            """,
            conn,
        )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "predicted_class", "brightness", "blur_score"])
    return df


def _uniform_class_distribution() -> Dict[str, float]:
    p = 1.0 / len(CLASS_NAMES)
    return {name: p for name in CLASS_NAMES}


def _normalize_distribution(counts: Dict[str, int], categories: List[str]) -> Dict[str, float]:
    total = sum(counts.get(cat, 0) for cat in categories)
    if total <= 0:
        return _uniform_class_distribution()
    return {cat: counts.get(cat, 0) / total for cat in categories}


def _stability_index(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _psi_from_class_distributions(expected_dist: Dict[str, float], actual_dist: Dict[str, float]) -> float:
    expected = np.array([expected_dist.get(c, 0.0) for c in CLASS_NAMES], dtype=float)
    actual = np.array([actual_dist.get(c, 0.0) for c in CLASS_NAMES], dtype=float)
    return _stability_index(expected, actual)


def _compute_bin_edges(reference_values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    reference_values = np.asarray(reference_values, dtype=float)
    if reference_values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference_values, quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        v = float(reference_values[0])
        edges = np.array([v - 0.5, v + 0.5], dtype=float)

    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6
    return edges


def _distribution_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    total = hist.sum()
    if total <= 0:
        return np.full(len(hist), 1.0 / len(hist), dtype=float)
    return hist / total


def _compute_rejection_rate(window_start: datetime, window_end: datetime) -> tuple:
    """Return (rejection_rate, reviewed_count) over prediction_logs in [start, end)."""
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE review_status = 'rejected') AS rejected,
                COUNT(*) FILTER (WHERE review_status IN ('approved', 'rejected')) AS reviewed
            FROM prediction_logs
            WHERE timestamp >= %s AND timestamp < %s
            """,
            (window_start, window_end),
        )
        rejected, reviewed = cur.fetchone()
    rejected = int(rejected or 0)
    reviewed = int(reviewed or 0)
    if reviewed == 0:
        return 0.0, 0
    return float(rejected) / float(reviewed), reviewed


def _csi_from_samples(expected_values: List[float], actual_values: List[float]) -> float:
    expected_values = np.asarray(expected_values, dtype=float)
    actual_values = np.asarray(actual_values, dtype=float)

    if expected_values.size == 0 or actual_values.size == 0:
        return 0.0

    edges = _compute_bin_edges(expected_values, n_bins=10)
    expected_dist = _distribution_from_edges(expected_values, edges)
    actual_dist = _distribution_from_edges(actual_values, edges)
    return _stability_index(expected_dist, actual_dist)


def bootstrap_baseline_task(**context):
    _ensure_dirs()

    if BASELINE_JSON.exists():
        logging.info("Baseline already exists at %s", BASELINE_JSON)
        return {"status": "exists", "path": str(BASELINE_JSON)}

    df = _read_logs()

    if df.empty:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "uniform_bootstrap_empty_logs",
            "class_distribution": _uniform_class_distribution(),
            "quality_reference": {
                "brightness": [128.0] * 50,
                "blur_score": [100.0] * 50,
            },
        }
    else:
        # Use earliest logs as the initial reference baseline
        df = df.sort_values("timestamp").head(500)
        counts = Counter(df["predicted_class"].astype(str).tolist())
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "bootstrap_from_prediction_logs",
            "class_distribution": _normalize_distribution(counts, CLASS_NAMES),
            "quality_reference": {
                "brightness": df["brightness"].astype(float).tolist(),
                "blur_score": df["blur_score"].astype(float).tolist(),
            },
        }

    with BASELINE_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logging.info("Created baseline at %s", BASELINE_JSON)
    return {"status": "created", "path": str(BASELINE_JSON)}


def compute_drift_task(**context):
    _ensure_dirs()

    if not BASELINE_JSON.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_JSON}")

    with BASELINE_JSON.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    df = _read_logs()
    if df.empty:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "empty_prediction_logs",
            "drift_detected": False,
            "psi": 0.0,
            "csi_brightness": 0.0,
            "csi_blur_score": 0.0,
            "threshold": DRIFT_THRESHOLD,
        }
        latest_path = REPORTS_DIR / "latest_report.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=WINDOW_DAYS)
    current_df = df[df["timestamp"] >= pd.Timestamp(window_start)]

    if len(current_df) < MIN_ROWS_FOR_DRIFT:
        report = {
            "generated_at": now.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": now.isoformat(),
            "status": "insufficient_recent_rows",
            "recent_rows": int(len(current_df)),
            "required_rows": MIN_ROWS_FOR_DRIFT,
            "drift_detected": False,
            "psi": 0.0,
            "csi_brightness": 0.0,
            "csi_blur_score": 0.0,
            "threshold": DRIFT_THRESHOLD,
        }
        latest_path = REPORTS_DIR / "latest_report.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    current_counts = Counter(current_df["predicted_class"].astype(str).tolist())
    current_class_dist = _normalize_distribution(current_counts, CLASS_NAMES)

    expected_class_dist = baseline.get("class_distribution", _uniform_class_distribution())
    psi = _psi_from_class_distributions(expected_class_dist, current_class_dist)

    expected_brightness = baseline.get("quality_reference", {}).get("brightness", [])
    expected_blur = baseline.get("quality_reference", {}).get("blur_score", [])

    current_brightness = current_df["brightness"].astype(float).tolist()
    current_blur = current_df["blur_score"].astype(float).tolist()

    csi_brightness = _csi_from_samples(expected_brightness, current_brightness)
    csi_blur = _csi_from_samples(expected_blur, current_blur)

    rejection_rate, reviewed_in_window = _compute_rejection_rate(window_start, now)

    reasons = []
    if psi > DRIFT_THRESHOLD:
        reasons.append("psi")
    if csi_brightness > DRIFT_THRESHOLD:
        reasons.append("csi_brightness")
    if csi_blur > DRIFT_THRESHOLD:
        reasons.append("csi_blur_score")
    if reviewed_in_window >= MIN_REVIEWED_IN_WINDOW and rejection_rate > REJECTION_RATE_THRESHOLD:
        reasons.append("rejection_rate")

    drift_detected = len(reasons) > 0

    report = {
        "generated_at": now.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "status": "ok",
        "threshold": DRIFT_THRESHOLD,
        "recent_rows": int(len(current_df)),
        "baseline_mode": baseline.get("mode", "unknown"),
        "psi": round(float(psi), 6),
        "csi_brightness": round(float(csi_brightness), 6),
        "csi_blur_score": round(float(csi_blur), 6),
        "rejection_rate": round(float(rejection_rate), 6),
        "reviewed_in_window": reviewed_in_window,
        "rejection_rate_threshold": REJECTION_RATE_THRESHOLD,
        "drift_detected": drift_detected,
        "reasons": reasons,
        "current_class_distribution": current_class_dist,
        "expected_class_distribution": expected_class_dist,
    }

    latest_path = REPORTS_DIR / "latest_report.json"
    timestamped_path = REPORTS_DIR / f"drift_report_{now.strftime('%Y%m%dT%H%M%SZ')}.json"

    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with timestamped_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info("Drift report saved to %s", latest_path)
    return report


def branch_on_drift(**context):
    report = context["ti"].xcom_pull(task_ids="compute_drift_report")
    if not (report and report.get("drift_detected")):
        logging.info("No drift detected; skipping retrain.")
        return "skip_model_retraining"

    # Persistence: drift must have fired in K of the last M reports
    recent_reports = sorted(REPORTS_DIR.glob("drift_report_*.json"))[-DRIFT_PERSISTENCE_WINDOW:]
    persisted_count = 0
    for path in recent_reports:
        try:
            with path.open("r", encoding="utf-8") as f:
                r = json.load(f)
            if r.get("drift_detected"):
                persisted_count += 1
        except Exception as exc:
            logging.warning(f"Could not read {path}: {exc}")

    if persisted_count < DRIFT_PERSISTENCE_K:
        logging.info(
            f"Drift detected this run but only {persisted_count}/{len(recent_reports)} "
            f"of the last {DRIFT_PERSISTENCE_WINDOW} runs had drift "
            f"(need {DRIFT_PERSISTENCE_K}). Skipping retrain (persistence)."
        )
        return "skip_model_retraining"

    # Cooldown: don't retrigger if model_training_pipeline has run recently
    try:
        from airflow.models import DagRun
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETRAIN_COOLDOWN_DAYS)
        runs = DagRun.find(dag_id="model_training_pipeline")
        for run in runs:
            sd = getattr(run, "start_date", None)
            if sd and sd >= cutoff:
                logging.info(
                    f"model_training_pipeline ran at {sd.isoformat()} "
                    f"(within {RETRAIN_COOLDOWN_DAYS}-day cooldown). Skipping retrain."
                )
                return "skip_model_retraining"
    except Exception as exc:
        logging.warning(f"Cooldown check failed; proceeding with retrain: {exc}")

    logging.info(
        f"Drift detected, persisted ({persisted_count}/{DRIFT_PERSISTENCE_WINDOW}), "
        f"cooldown cleared. Triggering retrain."
    )
    return "trigger_model_retraining"


default_args = {
    "owner": "ml_engineer",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id="drift_detection_pipeline",
    default_args=default_args,
    description="Detect production data drift using PSI/CSI",
    schedule="@daily",
    catchup=False,
    tags=["monitoring", "drift", "psi", "csi"],
) as dag:

    bootstrap_baseline = PythonOperator(
        task_id="bootstrap_baseline",
        python_callable=bootstrap_baseline_task,
    )

    compute_drift_report = PythonOperator(
        task_id="compute_drift_report",
        python_callable=compute_drift_task,
    )

    decide_retraining = BranchPythonOperator(
        task_id="decide_retraining",
        python_callable=branch_on_drift,
    )

    trigger_model_retraining = TriggerDagRunOperator(
        task_id="trigger_model_retraining",
        trigger_dag_id="model_training_pipeline",
        wait_for_completion=False,
        reset_dag_run=False,
    )

    skip_model_retraining = EmptyOperator(
        task_id="skip_model_retraining",
    )

    done = EmptyOperator(
        task_id="done",
        trigger_rule="none_failed_min_one_success",
    )

    bootstrap_baseline >> compute_drift_report >> decide_retraining
    decide_retraining >> trigger_model_retraining >> done
    decide_retraining >> skip_model_retraining >> done

