# C4 Code Level: ML Component (Airflow Pipelines Index)

## Overview

This document indexes the C4 Code-level documentation for the ML component of the SE-for-ML project, specifically the two Airflow DAGs that orchestrate model training and drift detection.

### ML Component Scope

The ML component consists of:

1. **Inference Pipeline** (`api/main.py` + `api/monitoring.py`)
   - See: `c4-code-ml-inference.puml`, `c4-code-ml-predict-sequence.puml`, `c4-code-ml.md`
   - Handles real-time predictions via FastAPI, prediction logging, review queue management

2. **Training Pipeline** (`airflow_pipeline/dags/train_pipeline.py`)
   - See: `c4-code-ml-training.puml`, `c4-code-ml-training.md`
   - Orchestrates feature extraction, 4-way model grid search, model promotion

3. **Drift Detection Pipeline** (`airflow_pipeline/dags/drift_detection_pipeline.py`)
   - See: `c4-code-ml-drift.puml`, `c4-code-ml-drift.md`
   - Monitors production data quality, triggers retraining on persistent drift

---

## Architecture Overview

### System Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│ Inference Service (api/)                                            │
├─────────────────────────────────────────────────────────────────────┤
│ main.py:predict()                                                   │
│  └─ EfficientNetV2-S backbone (torchvision, frozen)                │
│  └─ PCA(256) sklearn transformer                                   │
│  └─ LogisticRegression(6 classes) sklearn classifier              │
│  └─ write to prediction_logs (Postgres)                            │
│                                                                     │
│ monitoring.py:list_pending_reviews(), submit_review()             │
│  └─ read/update prediction_logs                                    │
│  └─ extract image_bytes, compute brightness & blur_score          │
│                                                                     │
│ auth.py: Session-based authentication (reviewer, user roles)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                       (contract: prediction_logs)
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Postgres: predictions-db (shared storage)                           │
├─────────────────────────────────────────────────────────────────────┤
│ prediction_logs table:                                              │
│  - request_id, timestamp, filename                                 │
│  - predicted_class, confidence_score, inference_time_ms            │
│  - brightness, blur_score, width, height                           │
│  - image_bytes (BYTEA), image_mime                                 │
│  - review_status ('pending'|'approved'|'rejected')                │
│  - final_class (corrected label), reviewed_at                      │
│                                                                     │
│ users table: username, password_hash, role ('user'|'reviewer')     │
└─────────────────────────────────────────────────────────────────────┘
          ↑                                              ↓
          └──────────────────┬───────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ↓                 ↓                 ↓
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Training DAG │  │ Drift DAG    │  │   MLflow     │
    │ (trigger)    │  │ (daily)      │  │   (S3-like)  │
    └──────────────┘  └──────────────┘  └──────────────┘
           ↓                 ↓                 ↓
    ┌──────────────────────────────────────────────────┐
    │ Docker Compose Stack (Airflow 3.0, scheduler)   │
    └──────────────────────────────────────────────────┘
```

### Key Data Flows

**A. Inference → Logging → Training**:
```
User prediction request
  → api/main.py:predict()
  → compute features (backbone)
  → predict via PCA + LogReg
  → append_prediction_log() → Postgres
    [monthly accumulation of logs + reviewer corrections]
    → train_pipeline.py:_load_reviewed_rows()
    → extract features from reviewed images
    → train 4-way grid (ResNet34/EfficientNetV2-S × train_v1/train_v2)
    → MLflow logs best model
    → current_model.json (promotion pointer)
```

**B. Inference → Monitoring → Drift Detection → Retraining**:
```
prediction_logs accumulates daily data
  → drift_detection_pipeline.py:compute_drift_task()
  → compute PSI, CSI, rejection_rate
  → branch_on_drift() gates via persistence + cooldown
    → if cleared: TriggerDagRunOperator → model_training_pipeline
    [retraining produces new current_model.json]
    → api/main.py:admin_reload() (reviewer-only endpoint)
    [API loads new model on next request or manual reload]
```

---

## File Locations & Artifacts

| File | Type | Purpose | Read By |
|---|---|---|---|
| `airflow_pipeline/dags/train_pipeline.py` | Source | Feature extraction, 4-way grid, model promotion | Airflow scheduler |
| `airflow_pipeline/dags/drift_detection_pipeline.py` | Source | Drift monitoring, retrain gating | Airflow scheduler (daily) |
| `airflow_pipeline/mlruns/current_model.json` | Data (JSON) | Model pointer (PCA + classifier paths, run_id, F1) | `api/main.py:load_model_pointer()` |
| `airflow_pipeline/mlruns/{exp_id}/models/{model_id}/artifacts/model.pkl` | Data (pickle) | Trained sklearn model (PCA or LogisticRegression) | `api/main.py:load_models()` via unpickling |
| `airflow_pipeline/data/drift/baseline_reference.json` | Data (JSON) | Reference class distribution + quality metrics | `drift_detection_pipeline.py:compute_drift_task()` |
| `airflow_pipeline/data/drift/reports/latest_report.json` | Data (JSON) | Current drift report (exposed via API) | `api/main.py:monitoring_drift_latest()` |
| `airflow_pipeline/data/drift/reports/drift_report_*.json` | Data (JSON, archived) | Timestamped drift reports for persistence gate | `drift_detection_pipeline.py:branch_on_drift()` |

---

## Class Taxonomy (Shared State)

**Critical**: CLASS_NAMES must be identical in 4 locations:

```python
CLASS_NAMES = [
    "beauty_salon",      # 0
    "drugstore",         # 1
    "restaurant",        # 2
    "movie_theater",     # 3
    "apartment_building", # 4
    "supermarket"        # 5
]
```

**Files**:
1. `api/main.py` (line 56): dict mapping index → name
2. `airflow_pipeline/dags/train_pipeline.py` (line 9): list of names
3. `airflow_pipeline/dags/drift_detection_pipeline.py` (line 28): list of names
4. `predict_test.py` (standalone script): dict mapping index → name

**Update Protocol**: If taxonomy changes, update all 4 locations + retrain.

---

## Inference Pipeline Reference

**See**: `c4-code-ml-inference.puml`, `c4-code-ml.md`

### Predict Endpoint (`POST /api/v1/predict`)

**Inputs**: Image file (JPG/PNG)  
**Outputs**: `{status, data: {predicted_industry, confidence_score, requires_manual_review, inference_time_ms}}`

**3-Stage Pipeline**:
1. **EfficientNetV2-S backbone**: Frozen pretrained, classifier → Identity (feature extraction)
2. **PCA(256)**: Dimensionality reduction (loaded from pickle)
3. **LogisticRegression**: 6-class classifier (loaded from pickle)

**Confidence Threshold**: 0.60
- If confidence < 0.60: `requires_manual_review = true`
- Prediction still returned; reviewer can approve or correct

**Prediction Logging** (`append_prediction_log`):
- Writes 16 columns to `prediction_logs` table (Postgres)
- Includes image_bytes for review queue
- Sets initial `review_status = 'pending'`

### Review Workflow

1. **Pending Items**: `GET /api/v1/review/pending` → FIFO list
2. **Get Image**: `GET /api/v1/review/{request_id}/image` → binary data
3. **Submit Review**: `POST /api/v1/review/{request_id}` (decision, final_class)
   - Sets `review_status` → 'approved' | 'rejected'
   - Sets `final_class` (if rejected)
   - Sets `reviewed_at` timestamp

Training pipeline reads approved/rejected rows and augments training set.

### Admin Reload

**Endpoint**: `POST /api/v1/admin/reload` (reviewer-only)
- Triggers `load_models()` → reads `current_model.json` → loads PCA + LogReg pickles
- Used after training DAG completes to load new model without restarting FastAPI

---

## Training Pipeline Reference

**See**: `c4-code-ml-training.puml`, `c4-code-ml-training.md`

### High-Level Flow

```
extract_features_task
  ├─ Load HF dataset: train_v1, train_v2, test
  ├─ Extract ResNet34 features → X_train_v1_resnet.npy, etc.
  ├─ Extract EfficientNetV2-S features → X_train_v1_effnet.npy, etc.
  ├─ Load reviewed images from Postgres prediction_logs
  └─ Extract reviewed features → X_reviewed_resnet.npy, X_reviewed_effnet.npy
         ↓
train_and_log_task
  ├─ 4-way grid: ResNet34 + train_v1, ResNet34 + train_v2,
  │             EfficientNetV2-S + train_v1, EfficientNetV2-S + train_v2
  ├─ For each config:
  │  ├─ Augment training set with reviewed samples
  │  ├─ Fit PCA(256) + LogisticRegression
  │  ├─ Evaluate on test set
  │  └─ Log to MLflow
  ├─ Select best test_f1
  └─ Write current_model.json (pca_pkl_path, clf_pkl_path, run_id, test_f1)
         ↓
regenerate_baseline_task
  └─ Delete baseline_reference.json → next drift run will re-bootstrap
```

### Feature Extraction Details

**HuggingFace Dataset**: `Punnarunwuwu/seml-industry-ver`
- Splits: train_v1, train_v2, test
- Each sample: {image (PIL), label (int), ...}
- Transform: Resize(224) → ToTensor → Normalize(ImageNet)

**Backbone Options**:
- ResNet34: `backbone.fc = Identity()` → 512-dim features
- EfficientNetV2-S: `backbone.classifier = Identity()` → 1280-dim features

**Memmap Output**: Features stored as memory-mapped `.npy` files (efficient I/O)

**Reviewed Samples**:
- Query `prediction_logs WHERE review_status IN ('approved', 'rejected')`
- Gate: Skip if < MIN_REVIEWED_FOR_RETRAIN (10) samples
- Decode image_bytes, apply same transform, extract features
- Save as X_reviewed_resnet.npy + X_reviewed_effnet.npy + y_reviewed.npy

### Model Training Details

**PCA + LogisticRegression Config**:
- PCA: 256 components, StandardScaler fit on training data
- LogReg: max_iter=2000, n_jobs=-1, C=1.0

**Grid Search**: 4 runs (all in one train_and_log_task)
```
ResNet34 + train_v1 → train_f1=0.85, test_f1=0.81
ResNet34 + train_v2 → train_f1=0.87, test_f2=0.82
EfficientNetV2-S + train_v1 → train_f1=0.88, test_f1=0.84 ← BEST
EfficientNetV2-S + train_v2 → train_f1=0.89, test_f1=0.83
```

**MLflow Logging**:
- Each run: params (backbone, dataset_split, components, reviewed_samples_added)
- Each run: metrics (train_accuracy, train_f1, test_accuracy, test_f1)
- Each run: models (pca_model/ and logistic_regression_model/)
- Summary run: F1 comparison bar chart

### Model Promotion

**Selection Criteria**: Highest test F1 (macro-averaged)

**Pointer Format** (current_model.json):
```json
{
  "pca_run_id": "abc123",
  "clf_run_id": "abc123",
  "pca_pkl_path": "airflow_pipeline/mlruns/1/models/m-uuid/artifacts/model.pkl",
  "clf_pkl_path": "airflow_pipeline/mlruns/1/models/m-uuid/artifacts/model.pkl",
  "winning_config": "EfficientNetV2-S / train_v1",
  "test_f1": 0.84,
  "promoted_at": "2026-04-23T12:34:56Z"
}
```

**Read by API**: `api/main.py:load_model_pointer()` on startup or reload

---

## Drift Detection Pipeline Reference

**See**: `c4-code-ml-drift.puml`, `c4-code-ml-drift.md`

### High-Level Flow

```
bootstrap_baseline_task
  ├─ If baseline_reference.json exists: no-op
  └─ Else: compute from earliest 500 prediction_logs
    {class_distribution, quality_reference: {brightness, blur_score}}
         ↓
compute_drift_task
  ├─ Read all prediction_logs
  ├─ Compute 4 signals over [now - WINDOW_DAYS, now):
  │  ├─ PSI: predicted-class distribution vs baseline
  │  ├─ CSI (brightness): brightness distribution vs baseline
  │  ├─ CSI (blur_score): blur_score distribution vs baseline
  │  └─ rejection_rate: % of reviewed predictions rejected
  ├─ drift_detected = any signal > DRIFT_THRESHOLD (0.25)
  └─ Save report to latest_report.json + timestamped file
         ↓
branch_on_drift
  ├─ Gate 1: If NOT drift_detected → skip retrain
  ├─ Gate 2: If < DRIFT_PERSISTENCE_K of last DRIFT_PERSISTENCE_WINDOW reports show drift → skip
  ├─ Gate 3: If model_training_pipeline ran within RETRAIN_COOLDOWN_DAYS → skip
  └─ Else: TriggerDagRunOperator(model_training_pipeline)
```

### Drift Signals

| Signal | Formula | Threshold | Meaning |
|---|---|---|---|
| **PSI** | Σ(actual - expected) × ln(actual/expected) | 0.25 | Predicted-class distribution shift |
| **CSI (brightness)** | Stability Index over histogram distributions | 0.25 | Image lighting condition shift |
| **CSI (blur)** | Stability Index over histogram distributions | 0.25 | Image sharpness/focus shift |
| **rejection_rate** | rejected / (approved + rejected) | 0.30 | % of reviews that are corrections |

**Interpretation**:
- PSI > 0.25 = significant distribution shift
- CSI > 0.25 = significant quality shift
- rejection_rate > 0.30 = >30% of predictions are wrong (per reviewer)
- `drift_detected = True` if any signal exceeds threshold

### Baseline Management

**Baseline Lifecycle**:
1. **Created**: On first drift DAG run if not exists (or after training)
2. **Refreshed**: Deleted by train_pipeline.py:regenerate_baseline_task()
3. **Re-created**: On next drift DAG run, bootstrapped from post-retrain predictions
4. **Used**: For all subsequent drift computations until training happens again

**Baseline Content**:
```json
{
  "created_at": "2026-04-23T12:34:56Z",
  "mode": "bootstrap_from_prediction_logs",
  "class_distribution": {"beauty_salon": 0.167, ...},
  "quality_reference": {
    "brightness": [103.5, 105.2, ...],
    "blur_score": [45.6, 48.2, ...]
  }
}
```

### Gating Strategy

**3-Layer Defense**:

1. **Drift Detection**: Any of 4 signals breaches threshold
2. **Persistence**: Requires K of M recent reports to show drift
   - Default: 3 of last 5 days must show drift_detected=true
   - **Prevents**: Retraining on one-off anomalies
3. **Cooldown**: model_training_pipeline must not have run recently
   - Default: 7-day cooldown
   - **Prevents**: Rapid cascades of expensive retraining

**Timeline Example**:
```
Day 1: drift=true, persisted=1/5 → skip (persistence)
Day 2: drift=true, persisted=2/5 → skip (persistence)
Day 3: drift=true, persisted=3/5 → trigger! (all gates pass)
       [model_training_pipeline starts]
Day 4: drift=true, persisted=4/5 → skip (cooldown: training running)
Day 5: drift=true, persisted=5/5 → skip (cooldown)
Day 10: drift=false → skip (no drift)
Day 11: drift=true, persisted=1/5 → skip (cooldown expired, but persistence low)
```

---

## Environment Configuration

### Training Pipeline Env Vars

| Variable | Default | Scope |
|---|---|---|
| `MIN_REVIEWED_FOR_RETRAIN` | 10 | Skip augmentation if fewer reviewed samples |
| `PREDICTIONS_DATABASE_URL` | `postgresql://predictions:predictions@predictions-db:5432/predictions` | Read reviewed rows |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |

### Drift Pipeline Env Vars

| Variable | Default | Scope |
|---|---|---|
| `DRIFT_THRESHOLD` | 0.25 | PSI and CSI thresholds |
| `DRIFT_WINDOW_DAYS` | 7 | Lookback window |
| `DRIFT_MIN_ROWS` | 20 | Minimum rows to compute drift |
| `REJECTION_RATE_THRESHOLD` | 0.30 | Rejection rate threshold |
| `MIN_REVIEWED_IN_WINDOW` | 10 | Gate for rejection_rate |
| `RETRAIN_COOLDOWN_DAYS` | 7 | Cooldown period |
| `DRIFT_PERSISTENCE_K` | 3 | Persistence requirement |
| `DRIFT_PERSISTENCE_WINDOW` | 5 | Persistence window size |

---

## Diagrams

### Training Pipeline Architecture
**File**: `c4-code-ml-training.puml`
- Shows: extract_features_task, train_and_log_task, regenerate_baseline_task
- Shows: class mappings, feature extraction flow, model grid search, MLflow integration
- Shows: current_model.json promotion

### Drift Detection Pipeline Architecture
**File**: `c4-code-ml-drift.puml`
- Shows: bootstrap_baseline_task, compute_drift_task, branch_on_drift decision
- Shows: 4 drift signals (PSI, CSI brightness, CSI blur, rejection_rate)
- Shows: Persistence and cooldown gates
- Shows: TriggerDagRunOperator to model_training_pipeline

### Inference Pipeline (Reference)
**File**: `c4-code-ml-inference.puml`
- Shows: FastAPI endpoints, 3-stage predict pipeline, prediction logging
- Shows: Review queue management, authentication roles

### Predict Sequence
**File**: `c4-code-ml-predict-sequence.puml`
- Shows: Temporal sequence of predict request → feature extraction → classification → logging

---

## Cross-References

| Concern | Location |
|---|---|
| Inference implementation | `c4-code-ml-inference.puml`, `c4-code-ml.md` |
| Training implementation | `c4-code-ml-training.puml`, `c4-code-ml-training.md` |
| Drift detection implementation | `c4-code-ml-drift.puml`, `c4-code-ml-drift.md` |
| Component-level overview | `c4-component-ml.puml`, `c4-component-ml.md` |
| System context | (parent C4 agent) |

---

## Key Takeaways

1. **Shared Contract**: `prediction_logs` table is the bridge between inference (writer) and training/monitoring (readers)
2. **Model Promotion**: `current_model.json` is the lightweight pointer (<<10KB) that enables rapid model updates without Docker restarts
3. **Two-Layer Training Integration**:
   - Automatic: Drift detection triggers via TriggerDagRunOperator
   - Manual: Reviewer uses `/api/v1/admin/reload` to activate new model
4. **Drift Gating**: Persistence + cooldown prevent resource waste; allow time for persistence to build signal strength
5. **Baseline Regeneration**: Key to preventing false drift signals after model updates; forces new reference baseline
6. **Shared State Risk**: CLASS_NAMES in 4 places; any change requires coordinated updates across codebase

---

## Next Steps

- Review detailed implementations in linked markdown files
- Check PlantUML diagrams for visual relationships
- Verify environment variables are set in docker-compose.yml
- Monitor drift reports (`latest_report.json`) via `/api/v1/monitoring/drift/latest` endpoint
