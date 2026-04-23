# C4 Code Level: ML Training Pipeline

## Overview

- **Name**: ML Model Training Pipeline (train_pipeline.py)
- **Description**: Airflow DAG that orchestrates 4-way grid search over ResNet34 and EfficientNetV2-S backbones paired with PCA+LogisticRegression classifiers. Integrates reviewer corrections from production logs, trains best-F1 model, and promotes it via `current_model.json` for inference API consumption.
- **Location**: `/Users/phudit/Downloads/Dev_Projects/SE-for-ML-Project/airflow_pipeline/dags/train_pipeline.py`
- **Language**: Python 3.10
- **Purpose**: Generate a production-ready classifier that incorporates both public training data (Hugging Face) and validated reviewer corrections from live predictions, with MLflow experiment tracking and model artifact management.

## Code Elements

### Configuration & Constants

#### CLASS_NAMES (line 9-16)
```python
CLASS_NAMES = [
    "beauty_salon", "drugstore", "restaurant", "movie_theater",
    "apartment_building", "supermarket"
]
CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
```
- **Type**: List[str] + Dict[str, int]
- **Purpose**: 6-class taxonomy shared with `api/main.py`, `drift_detection_pipeline.py`, and `predict_test.py`. Must be kept in sync across all four files.
- **Dependency**: Updates require synchronization across codebase

#### MIN_REVIEWED_FOR_RETRAIN (line 19)
- **Type**: int (env-controlled, default 10)
- **Purpose**: Gate for augmenting training set with reviewed production samples. If fewer than 10 approved/rejected predictions exist, training uses HF data only.

#### MLRUNS_CONTAINER_DIR, MLRUNS_HOST_DIR, CURRENT_MODEL_CONTAINER_PATH (lines 27-30)
- **Type**: Path (container) + str (host)
- **Purpose**: Paths to MLflow artifact store. Container path `/opt/airflow/mlruns` maps to host `airflow_pipeline/mlruns`.
- **Contract**: `current_model.json` written here; `api/main.py:load_model_pointer()` reads from host version.

#### PREDICTIONS_DATABASE_URL (line 21-24)
- **Type**: str (env-controlled)
- **Default**: `postgresql://predictions:predictions@predictions-db:5432/predictions`
- **Inside container**: Points to Docker service `predictions-db:5432`
- **Scope**: Reads reviewed prediction logs (image_bytes, final_class, review_status)

---

### Task 1: Feature Extraction (`extract_features_task`)

**Location**: line 114–320  
**Callable**: `PythonOperator(task_id="extract_features", python_callable=extract_features_task)`

#### extract_features_task() → None
- **Inputs**: None (reads from filesystem, HF, predictions-db, device)
- **Outputs**: 
  - `X_train_v1_resnet.npy`, `X_train_v2_resnet.npy`, `X_test_resnet.npy` (ResNet34 features)
  - `X_train_v1_effnet.npy`, `X_train_v2_effnet.npy`, `X_test_effnet.npy` (EfficientNetV2-S features)
  - `y_train_v1.npy`, `y_train_v2.npy`, `y_test.npy` (labels)
  - `X_reviewed_resnet.npy`, `X_reviewed_effnet.npy`, `y_reviewed.npy` (optional, reviewed samples)
- **Location**: `/opt/airflow/data/` inside container
- **Process**:
  1. Set reproducibility seeds (random, numpy, torch)
  2. Load HF dataset `Punnarunwuwu/seml-industry-ver` (train_v1, train_v2, test splits)
  3. Define transform: `Resize(224) → ToTensor → Normalize(ImageNet))`
  4. Load ResNet34 backbone (pretrained, replace `fc` with `Identity()`)
  5. Extract and save features using memmap for memory efficiency
  6. Load reviewed samples via `_load_reviewed_rows()` and extract their features
  7. Repeat for EfficientNetV2-S backbone
  8. Save all labels as numpy arrays

**Device Handling**: Auto-detects CUDA; falls back to CPU. Half-precision disabled in current config.

**Key Constraints**:
- `num_workers=0` in DataLoaders (no multiprocessing in Airflow workers)
- No `num_proc` in `load_dataset()` (would deadlock)

---

#### _load_reviewed_rows(transform, device) → (Tensor | None, ndarray | None)
**Location**: line 38–92

```python
def _load_reviewed_rows(transform, device):
    """Return (images_tensor, labels_array) for all reviewed prediction_logs rows.
    
    Returns (None, None) if below MIN_REVIEWED_FOR_RETRAIN.
    """
```

- **Parameters**:
  - `transform`: torchvision.transforms.Compose (same as used for HF dataset)
  - `device`: torch.device (cuda or cpu)
- **Returns**: Tuple[torch.Tensor, np.ndarray] | (None, None)
- **Process**:
  1. Query `prediction_logs` table: `WHERE review_status IN ('approved', 'rejected') AND final_class IS NOT NULL`
  2. Gate: if len(rows) < MIN_REVIEWED_FOR_RETRAIN (default 10), return (None, None)
  3. Decode image_bytes → PIL Image → apply transform
  4. Map final_class string → label index via CLASS_NAME_TO_INDEX
  5. Stack tensors and labels; move to device
- **Error Handling**: Logs warnings for undecodable images or unknown class names, continues processing

**Database Dependency**: Reads from `PREDICTIONS_DATABASE_URL` (predictions-db service inside Docker)

---

#### _extract_reviewed_features(model, x_tensor, batch_size=64, use_half=False) → ndarray
**Location**: line 95–108

- **Parameters**:
  - `model`: nn.Module (ResNet34 or EfficientNetV2-S with pooling layer replaced)
  - `x_tensor`: torch.Tensor of shape (N, 3, 224, 224)
  - `batch_size`: int (default 64)
  - `use_half`: bool (apply float16 conversion; default False)
- **Returns**: np.ndarray of shape (N, feature_dim) as float32
- **Process**: Batch inference through backbone; concatenate outputs
- **Memory**: Uses CPU for output to avoid GPU memory spike

---

### Task 2: Train & Evaluate Models (`train_and_log_task`)

**Location**: line 377–537  
**Callable**: `PythonOperator(task_id="train_and_evaluate_models", python_callable=train_and_log_task)`

#### train_and_log_task() → None
- **Inputs**: Feature arrays from Task 1 (`.npy` files)
- **Outputs**: 
  - Logs to MLflow (PCA model, LogisticRegression model, metrics, params)
  - **Primary**: `airflow_pipeline/mlruns/current_model.json` with winning model pointers
  - Visualization: `f1_comparison_chart.png` (bar chart of test F1 across 4 configs)
- **Process**:

1. **Load feature matrices**:
   ```python
   X_train_v1 = np.load("X_train_v1_resnet.npy")
   X_train_v2 = np.load("X_train_v2_resnet.npy")
   X_test = np.load("X_test_resnet.npy")
   y_train_v1, y_train_v2, y_test = [np.load(...) for ...in label files]
   ```

2. **Connect to MLflow**:
   ```python
   mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
   mlflow.set_experiment("Places365_Classification")
   ```

3. **Iterate 4-way grid**:
   ```
   models_to_train = [
       ResNet34 + train_v1,
       ResNet34 + train_v2,
       EfficientNetV2-S + train_v1,
       EfficientNetV2-S + train_v2
   ]
   ```

4. **For each config**:
   - Augment with reviewed samples: `_augment_with_reviewed(X_train, y_train, data_dir, reviewed_feat_file)` → (X_aug, y_aug, count_added)
   - Create MLflow run: `mlflow.start_run(run_name=f"PCA256_LogReg_{backbone}_{split}")`
   - Log params: pca_components=256, classifier, backbone, dataset_split, dataset URL, reviewed_samples_added
   - Fit PCA(256): `X_train_pca = pca.fit_transform(X_train_aug)`
   - Fit LogisticRegression: `clf.fit(X_train_pca, y_train_aug)` (max_iter=2000, n_jobs=-1, C=1.0)
   - Predict on train & test: `train_preds = clf.predict(X_train_pca)`, `test_preds = clf.predict(X_test_pca)`
   - Compute metrics: accuracy_score, f1_score (macro average)
   - Log metrics: train_accuracy, train_f1, test_accuracy, test_f1
   - Log models: `mlflow.sklearn.log_model(pca, "pca_model")`, `mlflow.sklearn.log_model(clf, "logistic_regression_model")`
   - Track if test_f1 > best_f1: update `best_record` with ModelInfo, run_id, config_name, test_f1

5. **Model Promotion** (if best_record exists):
   - Call `_resolve_pkl_path()` for both PCA and classifier ModelInfo objects
   - Construct `current_model.json` payload:
     ```json
     {
       "pca_run_id": "...",
       "clf_run_id": "...",
       "pca_pkl_path": "airflow_pipeline/mlruns/1/models/m-.../artifacts/model.pkl",
       "clf_pkl_path": "airflow_pipeline/mlruns/1/models/m-.../artifacts/model.pkl",
       "winning_config": "EfficientNetV2-S / train_v1",
       "test_f1": 0.8753,
       "promoted_at": "2026-04-23T12:34:56Z"
     }
     ```
   - Write to container path: `/opt/airflow/mlruns/current_model.json` (maps to host `airflow_pipeline/mlruns/current_model.json`)

6. **Visualization**:
   - Generate bar chart of test F1 across 4 configs
   - Save as `f1_comparison_chart.png`
   - Log to MLflow in a summary run

---

#### _augment_with_reviewed(X_train, y_train, data_dir, reviewed_feat_file) → (ndarray, ndarray, int)
**Location**: line 326–344

```python
def _augment_with_reviewed(X_train, y_train, data_dir, reviewed_feat_file):
    """Concatenate reviewed features + labels to the HF-derived arrays."""
```

- **Parameters**:
  - `X_train`: np.ndarray of HF-extracted features
  - `y_train`: np.ndarray of HF labels
  - `data_dir`: str (e.g., "/opt/airflow/data")
  - `reviewed_feat_file`: str (e.g., "X_reviewed_resnet.npy")
- **Returns**: (X_combined, y_combined, count_reviewed_added)
- **Process**:
  1. Check if `{data_dir}/{reviewed_feat_file}` and `{data_dir}/y_reviewed.npy` exist
  2. If not, return original (X_train, y_train, 0)
  3. Load reviewed features and labels
  4. If reviewed array is empty, return original
  5. Concatenate: `X_combined = np.concatenate([X_train, X_reviewed], axis=0)`
  6. Return count of reviewed samples added for logging

**Gate**: Only applied if both files exist (set by Task 1 when reviewed rows ≥ MIN_REVIEWED_FOR_RETRAIN)

---

#### _resolve_pkl_path(model_info) → str | None
**Location**: line 347–374

```python
def _resolve_pkl_path(model_info) -> str | None:
    """Return a HOST-relative path to the pickled model for current_model.json."""
```

- **Parameter**: `model_info` (return value from `mlflow.sklearn.log_model()`)
- **Returns**: str (HOST-relative path like `"airflow_pipeline/mlruns/1/models/m-.../artifacts/model.pkl"`) or None
- **Process** (MLflow 3 layout priority):
  1. Extract `model_info.model_id` → `m-<uuid>`
  2. Query MLflow run: `mlflow.get_run(model_info.run_id)` → extract `run.info.experiment_id`
  3. Compute path: `{MLRUNS_HOST_DIR}/{exp_id}/models/{model_id}/artifacts/model.pkl`
  4. **Fallback** (MLflow 2 legacy):
     - Parse `model_info.artifact_uri` (e.g., `file:///.../mlruns/1/<run>/artifacts/pca_model`)
     - Extract container path, strip `/opt/airflow/mlruns`, prepend `MLRUNS_HOST_DIR`
     - Return modified path with `/model.pkl` appended
  5. If both fail, return None

**Critical for API**: `api/main.py:load_model_pointer()` reads this JSON and loads the pickled models from these paths.

---

### Task 3: Regenerate Drift Baseline (`regenerate_baseline_task`)

**Location**: line 542–547  
**Callable**: `PythonOperator(task_id="regenerate_baseline", python_callable=regenerate_baseline_task)`

#### regenerate_baseline_task() → None
- **Purpose**: Reset drift detection baseline so the next drift DAG run re-bootstraps from post-retrain predictions
- **Process**:
  1. Check if `baseline_reference.json` exists at `/opt/airflow/data/drift/baseline_reference.json`
  2. If yes, delete it
  3. Log action
- **Effect**: Clears stale baseline; drift_detection_pipeline will compute fresh baseline on next run

---

### DAG Orchestration

**Location**: line 553–583

```python
with DAG(
    "model_training_pipeline",
    default_args={
        "owner": "ml_engineer",
        "depends_on_past": False,
        "start_date": datetime(2023, 1, 1),
        "retries": 0,
    },
    description="Training pipeline comparing ResNet34 and EfficientNetV2-S",
    schedule=None,  # Trigger-only (driven by drift_detection_pipeline)
    catchup=False,
    tags=["image_classification", "mlflow"],
) as dag:
    extract_task >> train_log_task >> regenerate_baseline
```

- **DAG ID**: `model_training_pipeline`
- **Schedule**: `None` (manually triggered by `drift_detection_pipeline.py` via `TriggerDagRunOperator`)
- **Task Dependency**: Linear chain: extract → train_log → regenerate

---

## Dependencies

### Internal Code Dependencies

| Dependency | Module | Purpose |
|---|---|---|
| `CLASS_NAMES` | `api/main.py` | Must match exactly for inference |
| `CLASS_NAMES` | `drift_detection_pipeline.py` | Must match for drift computations |
| `CLASS_NAMES` | `predict_test.py` | Must match for test script |
| `current_model.json` writer | `api/main.py:load_model_pointer()` | Consumes model pointers |
| Reviewed rows | `api/monitoring.py:submit_review()` | Source of training augmentation |

### External Dependencies

| Dependency | Version/URL | Purpose | Location |
|---|---|---|---|
| `torch` | PyTorch | Backbone models (ResNet34, EfficientNetV2-S) | extract_features_task |
| `torchvision` | TorchVision | Model architectures + transforms | extract_features_task |
| `sklearn` | scikit-learn | PCA, LogisticRegression | train_and_log_task |
| `numpy` | NumPy | Feature arrays, memmap | extract_features_task, train_and_log_task |
| `pandas` | Pandas | (imported but minimal use in train) | - |
| `mlflow` | MLflow | Experiment tracking, model logging | train_and_log_task |
| `datasets` | HF Datasets | Load `Punnarunwuwu/seml-industry-ver` | extract_features_task |
| `psycopg2` | psycopg2 | Query prediction_logs for reviewed samples | _load_reviewed_rows |
| `matplotlib`, `seaborn` | Optional | F1 comparison chart | train_and_log_task |
| `airflow` | Apache Airflow 3.0.0 | DAG orchestration, operators | DAG definition |
| HF Dataset | `Punnarunwuwu/seml-industry-ver` | train_v1, train_v2, test splits | extract_features_task |
| MLflow Server | `http://mlflow:5000` (Docker) | Track experiments, store artifacts | train_and_log_task |
| PostgreSQL (predictions-db) | `postgresql://predictions:predictions@predictions-db:5432/predictions` | Read reviewed rows | _load_reviewed_rows |

---

## Key Behavioral Patterns

### Reproducibility & Seeding
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```
All randomness locked to deterministic state for reproducible feature extraction.

### Model Promotion Strategy
1. Train all 4 configs
2. Compare test_f1 (macro-averaged)
3. Promote best as `current_model.json` pointer
4. API reloads on startup or via `/api/v1/admin/reload` (reviewer-only)

### Reviewed Data Augmentation Gate
- If `len(reviewed_samples) < MIN_REVIEWED_FOR_RETRAIN` (10): skip augmentation, use HF data only
- Logged in MLflow params: `reviewed_samples_added`
- Enables graceful degradation: training always succeeds even with few/no reviews

### Feature Extraction Efficiency
- **Memmap arrays**: Avoid loading full feature matrices into memory; use memory-mapped `.npy` files
- **GPU → CPU**: Features computed on device, immediately moved to CPU for storage
- **Batch processing**: Fixed batch_size=512 for consistent memory usage

### MLflow Integration Points
- **Experiment**: "Places365_Classification"
- **Per-run params**: backbone, dataset_split, PCA components, classifier hyperparams, reviewed_samples_added
- **Per-run metrics**: train_accuracy, train_f1, test_accuracy, test_f1
- **Per-run artifacts**: PCA model (pca_model/), LogisticRegression model (logistic_regression_model/)
- **Summary run**: F1 comparison visualization (bar chart PNG)

---

## Critical Data Contracts

### prediction_logs Table Schema
```sql
SELECT image_bytes, final_class, review_status
FROM prediction_logs
WHERE review_status IN ('approved', 'rejected')
```
- **image_bytes**: Raw image bytes (BYTEA)
- **final_class**: Reviewer-corrected class label (TEXT, one of CLASS_NAMES)
- **review_status**: 'pending' | 'approved' | 'rejected'

This is the only contract point with the inference API for training data.

### current_model.json Format
```json
{
  "pca_run_id": "...",
  "clf_run_id": "...",
  "pca_pkl_path": "airflow_pipeline/mlruns/1/models/m-.../artifacts/model.pkl",
  "clf_pkl_path": "airflow_pipeline/mlruns/1/models/m-.../artifacts/model.pkl",
  "winning_config": "EfficientNetV2-S / train_v1",
  "test_f1": 0.8753,
  "promoted_at": "2026-04-23T12:34:56Z"
}
```

Written by `train_and_log_task()` at:
- Container: `/opt/airflow/mlruns/current_model.json`
- Host: `airflow_pipeline/mlruns/current_model.json`

Read by `api/main.py:load_model_pointer()` on every API startup and reload.

---

## Fault Tolerance & Error Handling

| Failure Scenario | Handling |
|---|---|
| HF dataset unavailable | Exception propagates; task fails; Airflow retries (configured as 0 retries) |
| Reviewed row image decode fails | Logged as warning; row skipped; training continues if ≥ 0 valid rows |
| Unknown class in reviewed row | Logged as warning; row skipped; only valid classes used |
| MLflow connection fails | Exception propagates; task fails; no promotion occurs |
| model_info.artifact_uri parse fails | Fallback parsing attempted; if both fail, no promotion; error logged |
| CUDA out of memory | Falls back to CPU or reduces batch_size (not auto-implemented; manual config required) |

---

## Environment Variables

| Variable | Default | Scope |
|---|---|---|
| `MIN_REVIEWED_FOR_RETRAIN` | 10 | Gate for augmentation |
| `PREDICTIONS_DATABASE_URL` | `postgresql://predictions:predictions@predictions-db:5432/predictions` | PostgreSQL connection (inside Docker) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server |
| `HF_HUB_ENABLE_HF_TRANSFER` | 1 | Fast HF dataset downloads |
| `TOKENIZERS_PARALLELISM` | false | Avoid tokenizer warnings |

---

## Notes

1. **Shared STATE**: CLASS_NAMES must be kept in sync across `api/main.py`, `drift_detection_pipeline.py`, `train_pipeline.py`, and `predict_test.py`. Changes in one place require updates in all four.

2. **Model Pointer Fallback**: If `current_model.json` does not exist, `api/main.py` falls back to two hardcoded legacy artifact paths. This allows the system to start before the first successful training run.

3. **Drift Baseline Regeneration**: Deleting `baseline_reference.json` forces the drift DAG to re-bootstrap on the next run using fresh post-retrain predictions. This is critical after model updates to avoid false drift signals from reference shift.

4. **Docker Path Mapping**: Container paths (`/opt/airflow/mlruns`) map to host directory (`airflow_pipeline/mlruns`) via docker-compose volume mount. The `current_model.json` is the bridge: written in container, read on host by the API.

5. **No Distributed Training**: Current setup does not use distributed training (e.g., DataParallel, DistributedDataParallel). Single GPU or CPU only.
