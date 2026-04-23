# C4 Component Diagram — ML Subsystem

Source: [`c4-component-ml.puml`](./c4-component-ml.puml)

Uses [C4-PlantUML](https://github.com/plantuml-stdlib/C4-PlantUML) (official C4 macro library, latest `master`).

## How to render

**Option 1 — online (fastest for the slide deck):**
Open <https://www.plantuml.com/plantuml/uml/> and paste the contents of `c4-component-ml.puml`. Export as PNG or SVG.

**Option 2 — VS Code:**
Install the "PlantUML" extension (jebbs.plantuml), open the `.puml` file, `Alt+D` to preview, right-click preview → Export.

**Option 3 — CLI (Docker):**
```bash
docker run --rm -v "$PWD/C4-Documentation:/work" plantuml/plantuml \
  -tpng /work/c4-component-ml.puml
```

## Scope — ML component only

Per rubric "Component Diagram (ML component only) — internal structure of key containers" — this diagram shows ML-related components only. **Excluded on purpose**: auth, session management, review CRUD, user account management, static file serving — those are orthogonal to the ML subsystem.

Included:
- **FastAPI — ML Inference** container — the components that turn a request into a prediction.
- **Airflow DAG: `drift_detection_pipeline`** — the four daily drift tasks.
- **Airflow DAG: `model_training_pipeline`** — the three training tasks.
- Shared data stores that link them: `prediction_logs`, `current_model.json`, `baseline_reference.json`, MLflow store.

## Components

### FastAPI — ML Inference

| Component | Location | Role |
|---|---|---|
| `predict` endpoint | `api/main.py:predict` | `POST /api/v1/predict` — inference entry point |
| `admin_reload` endpoint | `api/main.py:admin_reload` | `POST /api/v1/admin/reload` — re-reads pointer, swaps models in memory |
| Model Pointer Loader | `api/main.py:load_model_pointer` | Reads `current_model.json`; falls back to legacy UUID paths if absent |
| Backbone | torchvision `efficientnet_v2_s` | Pretrained EfficientNetV2-S, classifier → `nn.Identity`, 1280-dim features |
| PCA | pickled sklearn | Reduces 1280 → 256 dims |
| Classifier | pickled sklearn `LogisticRegression` | Predicts class + softmax confidence (below 0.60 flags `requires_manual_review`) |
| Image-quality metrics | `api/monitoring.py:compute_brightness`, `:compute_blur_score` | Drift-monitoring features computed per request |
| Prediction Logger | `api/monitoring.py:append_prediction_log` | `INSERT` row into `prediction_logs` (image_bytes, predicted_class, features, etc.) |

### Airflow Scheduler — Drift DAG (`drift_detection_pipeline`)

| Task | Operator | Role |
|---|---|---|
| `bootstrap_baseline` | PythonOperator | Creates `baseline_reference.json` from earliest 500 rows if missing |
| `compute_drift_report` | PythonOperator | PSI + CSI(brightness) + CSI(blur) + rejection_rate over `DRIFT_WINDOW_DAYS` |
| `branch_on_drift` | BranchPythonOperator | Persistence (3-of-5 reports) + cooldown (7 days) gates |
| `trigger_model_retraining` | TriggerDagRunOperator | Fires `model_training_pipeline` DAG |

### Airflow Scheduler — Training DAG (`model_training_pipeline`)

| Task | Operator | Role |
|---|---|---|
| `extract_features_task` | PythonOperator | HF splits + reviewed rows → features for 4-way grid |
| `train_and_log_task` | PythonOperator + MLflow | Fits `PCA(256) + LogReg` per config, logs to MLflow, writes `current_model.json` with best-F1 paths |
| `regenerate_baseline_task` | PythonOperator | Deletes `baseline_reference.json` so next drift run re-bootstraps |

## Key relationships

- **Inference read path**: Endpoints → (Auth, Monitoring, Loader) → (Backbone → PCA → Classifier).
- **Drift read path**: `compute_drift_report` reads `prediction_logs` rolling window + `baseline_reference.json`, emits report JSON, `branch_on_drift` gates retrain.
- **Training write path**: `train_and_log_task` writes to MLflow and to `current_model.json`. `/admin/reload` makes the API re-read the pointer and swap models in memory.
- **Feedback loop**: Reviewer-labeled rows (`final_class`) in `prediction_logs` are pulled by `extract_features_task` and concatenated onto HF splits — retraining learns from corrections.
