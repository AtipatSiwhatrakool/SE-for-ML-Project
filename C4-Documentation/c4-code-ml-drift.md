# C4 Code Level: ML Drift Detection Pipeline

## Overview

- **Name**: ML Drift Detection & Retrain Trigger Pipeline (drift_detection_pipeline.py)
- **Description**: Airflow DAG that runs daily to detect data drift via PSI (predicted-class distribution), CSI (brightness and blur metrics), and rejection-rate signals. Incorporates two-layer gating (persistence + cooldown) before triggering model retraining to avoid cascading retrains on transient drift.
- **Location**: `/Users/phudit/Downloads/Dev_Projects/SE-for-ML-Project/airflow_pipeline/dags/drift_detection_pipeline.py`
- **Language**: Python 3.10
- **Purpose**: Monitor production prediction quality; detect systematic shifts in data distribution or reviewer disagreement; gate expensive model retraining with persistence and cooldown checks to maintain training pipeline stability.

## Code Elements

### Configuration & Constants

#### DriftConfig (lines 27–48)
```python
CLASS_NAMES = ["beauty_salon", "drugstore", "restaurant", "movie_theater", "apartment_building", "supermarket"]
DRIFT_THRESHOLD = 0.25  # float, env: DRIFT_THRESHOLD
WINDOW_DAYS = 7  # env: DRIFT_WINDOW_DAYS
MIN_ROWS_FOR_DRIFT = 20  # env: DRIFT_MIN_ROWS
REJECTION_RATE_THRESHOLD = 0.30  # env: REJECTION_RATE_THRESHOLD
MIN_REVIEWED_IN_WINDOW = 10  # env: MIN_REVIEWED_IN_WINDOW
RETRAIN_COOLDOWN_DAYS = 7  # env: RETRAIN_COOLDOWN_DAYS
DRIFT_PERSISTENCE_WINDOW = 5  # env: DRIFT_PERSISTENCE_WINDOW
DRIFT_PERSISTENCE_K = 3  # env: DRIFT_PERSISTENCE_K
```

- **CLASS_NAMES** (line 28–35): Must match `api/main.py`, `train_pipeline.py`, `predict_test.py`
- **DRIFT_THRESHOLD** (0.25): Threshold for PSI and CSI signals
- **WINDOW_DAYS** (7): Lookback window for computing drift signals
- **MIN_ROWS_FOR_DRIFT** (20): Minimum rows in window to compute drift; below this, report status "insufficient_recent_rows"
- **REJECTION_RATE_THRESHOLD** (0.30): If ≥30% of reviewed predictions are rejected, flag as drift signal
- **MIN_REVIEWED_IN_WINDOW** (10): Gate for rejection_rate computation; only compute if ≥10 reviews exist
- **RETRAIN_COOLDOWN_DAYS** (7): Don't trigger retraining if model_training_pipeline ran in last 7 days
- **DRIFT_PERSISTENCE_WINDOW** (5): Number of most recent timestamped drift reports to examine
- **DRIFT_PERSISTENCE_K** (3): Require ≥3 of last 5 reports to show drift before triggering retrain

#### DriftPaths (lines 18–20)
```python
DRIFT_DIR = Path("/opt/airflow/data/drift")
BASELINE_JSON = DRIFT_DIR / "baseline_reference.json"
REPORTS_DIR = DRIFT_DIR / "reports"
```

- **DRIFT_DIR**: Docker bind-mounted volume; persists across container restarts
- **BASELINE_JSON**: Reference class distribution and quality metrics (brightness, blur_score)
  - Created by `bootstrap_baseline_task()`
  - Deleted by `train_pipeline.py:regenerate_baseline_task()` after successful training
  - Re-bootstrapped from post-retrain predictions on next drift run
- **REPORTS_DIR**: Contains timestamped `drift_report_*.json` and `latest_report.json`

#### DatabaseUrl (lines 22–25)
```python
PREDICTIONS_DATABASE_URL = "postgresql://predictions:predictions@predictions-db:5432/predictions"
```

- **Inside container**: Points to Docker service `predictions-db`
- **Reads from**: `prediction_logs` table (timestamp, predicted_class, brightness, blur_score, review_status)

---

### Task 1: Bootstrap Baseline (`bootstrap_baseline_task`)

**Location**: line 161–198  
**Callable**: `PythonOperator(task_id="bootstrap_baseline", python_callable=bootstrap_baseline_task)`

#### bootstrap_baseline_task(**context) → Dict[str, Any]
```python
def bootstrap_baseline_task(**context):
    """Initialize or return status of baseline_reference.json."""
```

- **Returns**: `{"status": "created" | "exists", "path": str}`
- **Process**:
  1. Call `_ensure_dirs()` to create DRIFT_DIR, REPORTS_DIR
  2. Check if BASELINE_JSON already exists
     - If yes: return `{"status": "exists", "path": "..."}`
     - If no: proceed to bootstrap
  3. Read all prediction logs: `_read_logs()` → pd.DataFrame
  4. **If df is empty** (no predictions yet):
     - Create uniform class distribution: `{class: 1/6 for each class}`
     - Create dummy quality reference: `brightness: [128.0]*50, blur_score: [100.0]*50`
  5. **Else** (df has data):
     - Sort by timestamp, take first 500 rows
     - Count predicted_class occurrences
     - Normalize to class distribution via `_normalize_distribution()`
     - Extract quality metrics: brightness and blur_score as lists
  6. Write baseline payload to BASELINE_JSON:
     ```json
     {
       "created_at": "2026-04-23T12:34:56Z",
       "mode": "bootstrap_from_prediction_logs" | "uniform_bootstrap_empty_logs",
       "class_distribution": {"beauty_salon": 0.167, ...},
       "quality_reference": {
         "brightness": [103.5, 105.2, ...],
         "blur_score": [45.6, 48.2, ...]
       }
     }
     ```

**Key Behavior**: Baseline is created once and reused across drift detection runs. Deleted only by training pipeline after successful retrain.

---

### Task 2: Compute Drift Report (`compute_drift_task`)

**Location**: line 201–308  
**Callable**: `PythonOperator(task_id="compute_drift_report", python_callable=compute_drift_task)`

#### compute_drift_task(**context) → Dict[str, Any]
```python
def compute_drift_task(**context):
    """Compute PSI, CSI, rejection_rate; generate drift report."""
```

- **Returns**: Drift report dictionary (persisted to JSON file)
- **Outputs**:
  - `latest_report.json` (always overwritten)
  - `drift_report_YYYYMMDDTHHMMSSZ.json` (timestamped, archived)

**Process**:

1. **Load baseline** from BASELINE_JSON:
   ```python
   baseline = {"mode": "...", "class_distribution": {...}, "quality_reference": {...}}
   ```

2. **Read prediction logs** via `_read_logs()`:
   ```python
   df = pd.read_sql_query(SELECT ... FROM prediction_logs ORDER BY timestamp ASC)
   ```
   Columns: request_id, timestamp, filename, predicted_class, brightness, blur_score, width, height

3. **Check data availability**:
   - If df is empty: return status "empty_prediction_logs", drift_detected=False
   - Compute window: `[now - WINDOW_DAYS, now)`
   - If window has < MIN_ROWS_FOR_DRIFT (20): return status "insufficient_recent_rows", drift_detected=False

4. **Compute drift signals** over window:

   **a) PSI (Predicted-class distribution)**:
   ```python
   expected_dist = baseline.get("class_distribution")  # from bootstrap
   actual_dist = _normalize_distribution(Counter(current_df["predicted_class"]))
   psi = _psi_from_class_distributions(expected_dist, actual_dist)
   ```
   - Measures distribution shift in model predictions
   - High PSI = model output distribution has changed significantly

   **b) CSI (Brightness)**:
   ```python
   expected_brightness = baseline.get("quality_reference", {}).get("brightness")  # reference values
   current_brightness = current_df["brightness"].tolist()  # window values
   csi_brightness = _csi_from_samples(expected_brightness, current_brightness)
   ```
   - Measures shift in image lighting conditions
   - High CSI = images are much brighter/darker than baseline

   **c) CSI (Blur Score)**:
   ```python
   expected_blur = baseline.get("quality_reference", {}).get("blur_score")
   current_blur = current_df["blur_score"].tolist()
   csi_blur = _csi_from_samples(expected_blur, current_blur)
   ```
   - Measures shift in image sharpness
   - High CSI = images are much sharper/blurrier than baseline

   **d) Rejection Rate** (if ≥ MIN_REVIEWED_IN_WINDOW reviews):
   ```python
   rejection_rate, reviewed_count = _compute_rejection_rate(window_start, now)
   if reviewed_count >= MIN_REVIEWED_IN_WINDOW and rejection_rate > REJECTION_RATE_THRESHOLD:
       reasons.append("rejection_rate")
   ```
   - Measures reviewer disagreement rate: rejected / (approved + rejected)
   - High rejection_rate = reviewers are correcting many predictions

5. **Determine drift_detected**:
   ```python
   reasons = []
   if psi > DRIFT_THRESHOLD: reasons.append("psi")
   if csi_brightness > DRIFT_THRESHOLD: reasons.append("csi_brightness")
   if csi_blur > DRIFT_THRESHOLD: reasons.append("csi_blur_score")
   if reviewed_count >= MIN_REVIEWED_IN_WINDOW and rejection_rate > REJECTION_RATE_THRESHOLD:
       reasons.append("rejection_rate")
   drift_detected = len(reasons) > 0
   ```
   - `drift_detected = True` if **any** signal exceeds threshold

6. **Generate report** and save to both `latest_report.json` and timestamped file:
   ```json
   {
     "generated_at": "2026-04-23T12:34:56.123456Z",
     "window_start": "2026-04-16T12:34:56Z",
     "window_end": "2026-04-23T12:34:56Z",
     "status": "ok" | "empty_prediction_logs" | "insufficient_recent_rows",
     "threshold": 0.25,
     "recent_rows": 150,
     "baseline_mode": "bootstrap_from_prediction_logs",
     "psi": 0.18,
     "csi_brightness": 0.09,
     "csi_blur_score": 0.31,
     "rejection_rate": 0.35,
     "reviewed_in_window": 15,
     "rejection_rate_threshold": 0.30,
     "drift_detected": true,
     "reasons": ["csi_blur_score", "rejection_rate"],
     "current_class_distribution": {"beauty_salon": 0.12, ...},
     "expected_class_distribution": {"beauty_salon": 0.167, ...}
   }
   ```

---

### Data Preparation Functions

#### _read_logs() → pd.DataFrame
**Location**: line 56–74

```python
def _read_logs() -> pd.DataFrame:
    """Read all prediction_logs, parse timestamps, drop nulls."""
```

- **Query**:
  ```sql
  SELECT request_id, timestamp, filename, predicted_class,
         confidence_score, requires_manual_review, inference_time_ms,
         brightness, blur_score, width, height
  FROM prediction_logs
  ORDER BY timestamp ASC
  ```
- **Cleaning**:
  - Parse `timestamp` as UTC datetime
  - Drop rows with null: timestamp, predicted_class, brightness, blur_score
- **Returns**: Cleaned pd.DataFrame sorted by timestamp

#### _uniform_class_distribution() → Dict[str, float]
**Location**: line 77–79

- **Returns**: `{class: 1/6 for each class in CLASS_NAMES}`
- **Used for**: Empty logs baseline, fallback

#### _normalize_distribution(counts: Dict, categories: List) → Dict[str, float]
**Location**: line 82–86

- **Parameters**:
  - `counts`: Dict[str, int] (e.g., Counter result)
  - `categories`: List[str] (CLASS_NAMES)
- **Returns**: Normalized probabilities
- **Fallback**: If total ≤ 0, return uniform distribution

---

### Stability Index (PSI/CSI) Computation

#### _stability_index(expected: ndarray, actual: ndarray, eps: float = 1e-6) → float
**Location**: line 89–92

```python
def _stability_index(expected, actual, eps=1e-6) -> float:
    """Compute KL-divergence-like measure: sum((actual - expected) * log(actual / expected))."""
```

- **Formula**: Σ(actual - expected) × ln(actual / expected)
- **Numerical safety**: Clip values to [eps, ∞) to avoid log(0)
- **Interpretation**: 
  - 0 = no drift
  - > 0.25 = significant drift (typical threshold)

#### _psi_from_class_distributions(expected_dist: Dict, actual_dist: Dict) → float
**Location**: line 95–98

- **Inputs**: Both dicts with CLASS_NAMES keys
- **Process**:
  1. Extract probabilities in CLASS_NAMES order as numpy arrays
  2. Call `_stability_index(expected, actual)`
- **Returns**: PSI score
- **Use case**: Measure predicted-class distribution shift

#### _compute_bin_edges(reference_values: ndarray, n_bins: int = 10) → ndarray
**Location**: line 101–116

- **Purpose**: Compute histogram bins based on reference data quantiles
- **Process**:
  1. Compute percentiles: [0, 10, 20, ..., 90, 100]
  2. Extract corresponding values from reference_values
  3. Make edges unique; pad first/last by ±1e-6
- **Returns**: n_bins+1 edges
- **Use case**: Create consistent bins for CSI computation

#### _distribution_from_edges(values: ndarray, edges: ndarray) → ndarray
**Location**: line 119–124

- **Process**:
  1. Compute histogram(values, bins=edges)
  2. Normalize to probabilities
  3. Fallback to uniform if total ≤ 0
- **Returns**: np.ndarray of probabilities
- **Use case**: Convert sample values to histogram distribution

#### _csi_from_samples(expected_values: List[float], actual_values: List[float]) → float
**Location**: line 148–158

```python
def _csi_from_samples(expected_values, actual_values) -> float:
    """Compute CSI: Stability Index over histogram distributions."""
```

- **Process**:
  1. Compute bin edges from expected_values
  2. Compute expected distribution via histogram
  3. Compute actual distribution via histogram (same bins)
  4. Call `_stability_index(expected_dist, actual_dist)`
- **Returns**: CSI score
- **Use cases**: Measure brightness or blur_score distribution shift

---

### Rejection Rate Computation

#### _compute_rejection_rate(window_start: datetime, window_end: datetime) → (float, int)
**Location**: line 127–145

```python
def _compute_rejection_rate(window_start, window_end) -> tuple:
    """Return (rejection_rate, reviewed_count) over prediction_logs in [start, end)."""
```

- **Query**:
  ```sql
  SELECT
      COUNT(*) FILTER (WHERE review_status = 'rejected') AS rejected,
      COUNT(*) FILTER (WHERE review_status IN ('approved', 'rejected')) AS reviewed
  FROM prediction_logs
  WHERE timestamp >= window_start AND timestamp < window_end
  ```
- **Returns**:
  - `(0.0, 0)` if reviewed_count == 0
  - `(rejected / reviewed, reviewed_count)` otherwise
- **Interpretation**:
  - 0.0 = 0% rejection rate (all approved)
  - 0.5 = 50% rejection rate (half rejected)
  - 1.0 = 100% rejection rate (all rejected, invalid)
  - Only computed if reviewed_count ≥ MIN_REVIEWED_IN_WINDOW (10)

---

### Task 3: Branch on Drift (`branch_on_drift`)

**Location**: line 311–357  
**Callable**: `BranchPythonOperator(task_id="decide_retraining", python_callable=branch_on_drift)`

#### branch_on_drift(**context) → str
```python
def branch_on_drift(**context) -> str:
    """Decide whether to trigger model retraining based on drift + persistence + cooldown."""
```

- **Returns**: Task name to branch to:
  - `"trigger_model_retraining"` (retrain)
  - `"skip_model_retraining"` (skip)

**Decision Logic** (sequential gates, all must pass):

1. **Drift Detection Gate**:
   ```python
   report = context["ti"].xcom_pull(task_ids="compute_drift_report")
   if not report.get("drift_detected"):
       return "skip_model_retraining"
   ```
   - If no drift detected, skip immediately

2. **Persistence Gate**:
   ```python
   recent_reports = sorted(REPORTS_DIR.glob("drift_report_*.json"))[-DRIFT_PERSISTENCE_WINDOW:]
   persisted_count = sum(1 for path in recent_reports if json.load(path).get("drift_detected"))
   if persisted_count < DRIFT_PERSISTENCE_K:
       logging.info(f"Only {persisted_count}/{len(recent_reports)} of last {DRIFT_PERSISTENCE_WINDOW} "
                   f"showed drift; need {DRIFT_PERSISTENCE_K}. Skipping retrain.")
       return "skip_model_retraining"
   ```
   - Requires drift in ≥ K of the last M reports
   - Default: 3 of last 5 daily runs
   - **Rationale**: Prevents retraining on one-off anomalies; requires consistent drift signal

3. **Cooldown Gate**:
   ```python
   from airflow.models import DagRun
   cutoff = datetime.now(timezone.utc) - timedelta(days=RETRAIN_COOLDOWN_DAYS)
   runs = DagRun.find(dag_id="model_training_pipeline")
   for run in runs:
       if run.start_date >= cutoff:
           logging.info(f"model_training_pipeline ran at {run.start_date} "
                       f"(within {RETRAIN_COOLDOWN_DAYS}-day cooldown). Skipping retrain.")
           return "skip_model_retraining"
   ```
   - Check Airflow metadata: when did `model_training_pipeline` last run?
   - If within cooldown period (7 days), skip retrain
   - **Rationale**: Avoid expensive retraining if a recent training run already exists

4. **All gates passed**:
   ```python
   logging.info(f"Drift detected, persisted ({persisted_count}/{DRIFT_PERSISTENCE_WINDOW}), "
               f"cooldown cleared. Triggering retrain.")
   return "trigger_model_retraining"
   ```

**Example Timeline**:
```
Day 1: drift_detected=True, persisted=1/5 → skip (persistence gate)
Day 2: drift_detected=True, persisted=2/5 → skip (persistence gate)
Day 3: drift_detected=True, persisted=3/5 → trigger! (all gates pass)
       → model_training_pipeline starts (takes ~2 hours)
Day 4: drift_detected=True, persisted=4/5 → skip (cooldown gate: recent training)
Day 5: drift_detected=True, persisted=5/5 → skip (cooldown gate)
Day 10: drift_detected=False → skip (drift gate)
Day 11: drift_detected=True, persisted=1/5 → skip (cooldown expired, but persistence too low)
```

---

### DAG Orchestration

**Location**: line 360–410

```python
with DAG(
    dag_id="drift_detection_pipeline",
    default_args={
        "owner": "ml_engineer",
        "depends_on_past": False,
        "start_date": datetime(2023, 1, 1),
        "retries": 0,
    },
    description="Detect production data drift using PSI/CSI",
    schedule="@daily",  # Runs daily at 00:00 UTC
    catchup=False,
    tags=["monitoring", "drift", "psi", "csi"],
) as dag:
```

**Task Graph**:
```
bootstrap_baseline
    ↓
compute_drift_report
    ↓
decide_retraining
    ├─→ trigger_model_retraining → done
    └─→ skip_model_retraining ────→ done
```

**Key Tasks**:

| Task ID | Operator | Callable | Branches? |
|---|---|---|---|
| bootstrap_baseline | PythonOperator | bootstrap_baseline_task | No |
| compute_drift_report | PythonOperator | compute_drift_task | No |
| decide_retraining | BranchPythonOperator | branch_on_drift | Yes (2 branches) |
| trigger_model_retraining | TriggerDagRunOperator | N/A | Triggers `model_training_pipeline` |
| skip_model_retraining | EmptyOperator | N/A | No-op |
| done | EmptyOperator (trigger_rule="none_failed_min_one_success") | N/A | Merge both branches |

---

## Dependencies

### Internal Code Dependencies

| Dependency | Source | Purpose |
|---|---|---|
| `CLASS_NAMES` | `api/main.py`, `train_pipeline.py`, `predict_test.py` | Must be identical for label consistency |
| Baseline file | `train_pipeline.py:regenerate_baseline_task()` | Deletes baseline after retrain; drift DAG re-bootstraps |
| Model training DAG | Airflow metadata (`DagRun`) | Cooldown gate queries model_training_pipeline run history |
| Drift reports | File system (DRIFT_DIR/reports) | Persistence gate reads timestamped reports |

### External Dependencies

| Dependency | Version | Purpose | Location |
|---|---|---|---|
| `numpy` | NumPy | Array operations, percentile, histogram | Stability index, bin edge computation |
| `pandas` | Pandas | DataFrame manipulation, SQL query results | _read_logs |
| `psycopg2` | psycopg2 | PostgreSQL connection | _compute_rejection_rate, _read_logs |
| `json` | Built-in | Parse/write drift reports | compute_drift_task, branch_on_drift |
| `pathlib.Path` | Built-in | File system operations | DRIFT_DIR, REPORTS_DIR |
| `datetime, timezone` | Built-in | UTC timestamps, time calculations | Window computation, report timestamps |
| `airflow` | Apache Airflow 3.0.0 | DAG orchestration, operators, metadata | DAG definition, DagRun queries |
| PostgreSQL (predictions-db) | `postgresql://...@predictions-db:5432/predictions` | Read prediction_logs table | _read_logs, _compute_rejection_rate |

---

## Key Behavioral Patterns

### Daily Cadence
- **Schedule**: `@daily` (runs at 00:00 UTC by default in Airflow; container timezone should be UTC)
- **Lookback**: Always looks at `[now - WINDOW_DAYS, now)` regardless of when task runs
- **Reports**: Timestamped with ISO 8601 UTC to enable persistence gate (file-based ordering)

### Two-Layer Gating Strategy
```
Drift Detection (signal-based) → Persistence (temporal) → Cooldown (resource)
```
- **Layer 1**: If any of 4 signals breaches threshold, `drift_detected = true`
- **Layer 2**: Persistence ensures signal is sustained over multiple runs
- **Layer 3**: Cooldown prevents rapid cascades when model is already being retrained

**Rationale**: High-precision triggering for expensive retrains; avoid waste on transient anomalies.

### Baseline Regeneration Loop
```
Train DAG runs → regenerate_baseline_task() deletes baseline
    ↓
Drift DAG runs → bootstrap_baseline_task() creates new baseline from post-retrain predictions
    ↓
(Baseline used for next N drift runs until another train happens)
```

This ensures the baseline always represents the most recent training distribution, preventing stale reference drift.

### Four Drift Signals
1. **PSI (Predicted-class distribution)**: Monitors if model output distribution has shifted
2. **CSI (Brightness)**: Monitors if images are becoming brighter/darker (lighting conditions)
3. **CSI (Blur score)**: Monitors if images are becoming sharper/blurrier (focus/resolution)
4. **Rejection rate**: Monitors reviewer disagreement (correction rate)

Each is independent; any one exceeding threshold triggers drift. Combined, they cover:
- Model distribution shift (PSI)
- Input quality shift (CSI brightness/blur)
- Ground truth shift (rejection rate)

### File-Based Persistence
Persistence gate reads timestamped `drift_report_*.json` files from disk:
```python
recent_reports = sorted(REPORTS_DIR.glob("drift_report_*.json"))[-DRIFT_PERSISTENCE_WINDOW:]
```
- **No database**: Simplifies deployment; reports are artifacts
- **Ordering by filename**: Lexicographic sort == chronological order (ISO 8601 format: `YYYYMMDDTHHMMSSZ`)
- **Robustness**: If one report is corrupted, others are still readable

---

## Critical Data Contracts

### prediction_logs Schema
```sql
SELECT timestamp, predicted_class, brightness, blur_score, 
       review_status, final_class
FROM prediction_logs
WHERE timestamp >= window_start AND timestamp < window_end
```

Required columns:
- `timestamp` (TIMESTAMPTZ): Query window boundary
- `predicted_class` (TEXT): For PSI computation
- `brightness` (DOUBLE PRECISION): For brightness CSI
- `blur_score` (DOUBLE PRECISION): For blur CSI
- `review_status` ('pending'|'approved'|'rejected'): For rejection rate
- `final_class` (TEXT): Metadata; not used in drift computation

### baseline_reference.json Format
```json
{
  "created_at": "2026-04-23T12:34:56Z",
  "mode": "bootstrap_from_prediction_logs",
  "class_distribution": {
    "beauty_salon": 0.167,
    "drugstore": 0.167,
    ...
  },
  "quality_reference": {
    "brightness": [103.5, 105.2, 102.8, ...],
    "blur_score": [45.6, 48.2, 42.1, ...]
  }
}
```

Created by `bootstrap_baseline_task()`; deleted by training DAG; re-created by next drift run.

### drift_report_*.json Format
```json
{
  "generated_at": "2026-04-23T12:34:56Z",
  "window_start": "2026-04-16T12:34:56Z",
  "window_end": "2026-04-23T12:34:56Z",
  "status": "ok",
  "threshold": 0.25,
  "recent_rows": 150,
  "baseline_mode": "bootstrap_from_prediction_logs",
  "psi": 0.18,
  "csi_brightness": 0.09,
  "csi_blur_score": 0.31,
  "rejection_rate": 0.35,
  "reviewed_in_window": 15,
  "rejection_rate_threshold": 0.30,
  "drift_detected": true,
  "reasons": ["csi_blur_score", "rejection_rate"],
  "current_class_distribution": {...},
  "expected_class_distribution": {...}
}
```

Written to both `latest_report.json` (current state) and timestamped file (archive).

---

## Fault Tolerance & Error Handling

| Failure Scenario | Handling |
|---|---|
| PostgreSQL unavailable | Exception propagates; task fails; Airflow retries (0 retries configured) |
| prediction_logs table empty | Report status "empty_prediction_logs", drift_detected=False |
| Window has < MIN_ROWS_FOR_DRIFT | Report status "insufficient_recent_rows", drift_detected=False |
| Baseline file missing | compute_drift_task raises FileNotFoundError; task fails |
| Corrupted drift_report_*.json | Logged as warning; file skipped; count unchanged (persistence gate lenient) |
| Cooldown check fails (DagRun.find error) | Logged as warning; proceed with retrain (fail-open: safety to prevent stale model) |
| NaN in PSI/CSI computation | Numeric library clips to [eps, ∞) to avoid log(0); result is safe |

---

## Environment Variables

| Variable | Default | Scope |
|---|---|---|
| `DRIFT_THRESHOLD` | 0.25 | All signal thresholds (PSI, CSI) |
| `DRIFT_WINDOW_DAYS` | 7 | Lookback window for signals |
| `DRIFT_MIN_ROWS` | 20 | Minimum rows to compute signals |
| `REJECTION_RATE_THRESHOLD` | 0.30 | Rejection rate signal threshold |
| `MIN_REVIEWED_IN_WINDOW` | 10 | Gate for rejection_rate computation |
| `RETRAIN_COOLDOWN_DAYS` | 7 | Days to wait before next retrain |
| `DRIFT_PERSISTENCE_K` | 3 | Required count for persistence gate |
| `DRIFT_PERSISTENCE_WINDOW` | 5 | Window size for persistence gate |
| `PREDICTIONS_DATABASE_URL` | `postgresql://...@predictions-db:5432/predictions` | PostgreSQL connection |

All except the last are designed for tuning; start with defaults and adjust based on observed drift patterns.

---

## Notes

1. **Shared STATE**: CLASS_NAMES must match across `api/main.py`, `train_pipeline.py`, and `predict_test.py`. Any class taxonomy change requires coordinated updates.

2. **Time Zone**: All timestamps are UTC (ISO 8601 with Z suffix). Ensure Airflow scheduler and container time are synchronized to UTC.

3. **Drift Reports as Audit Trail**: Timestamped reports create an immutable record of drift signals; useful for post-mortems and understanding why retraining was triggered.

4. **Persistence Gate Tuning**: 
   - Aggressive (low K): More frequent retraining, higher cost
   - Conservative (high K): Fewer retrains, higher lag, risk of stale model
   - Default (3 of 5) = ~2 weeks required to override initial false positives

5. **Cooldown Period**:
   - Prevents Airflow from queueing multiple expensive training runs
   - If training takes ~2 hours and cooldown is 7 days, subsequent drift signals are ignored
   - After cooldown expires, system can retrain again

6. **Baseline Lifecycle**:
   - Created: On first drift DAG run if no baseline exists
   - Refreshed: Immediately after successful model retraining (train DAG deletes it)
   - Used: For next N drift computations until training happens again
   - Key insight: Baseline always represents post-retrain distribution, not pre-retrain

7. **XCom Dependency**: `branch_on_drift` pulls compute_drift_task results from Airflow XCom (task communication). If XCom is unavailable, branch_on_drift will fail.
