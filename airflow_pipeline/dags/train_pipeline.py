import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging


# Kept in sync with api/main.py:CLASS_NAMES and drift_detection_pipeline.py:CLASS_NAMES.
CLASS_NAMES = [
    "beauty_salon",
    "drugstore",
    "restaurant",
    "movie_theater",
    "apartment_building",
    "supermarket",
]
CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MIN_REVIEWED_FOR_RETRAIN = int(os.getenv("MIN_REVIEWED_FOR_RETRAIN", "10"))

PREDICTIONS_DATABASE_URL = os.getenv(
    "PREDICTIONS_DATABASE_URL",
    "postgresql://predictions:predictions@predictions-db:5432/predictions",
)

# Container path. Mounted from ./mlruns on host (docker-compose.yml).
MLRUNS_CONTAINER_DIR = "/opt/airflow/mlruns"
# Host-relative path the API resolves from repo root.
MLRUNS_HOST_DIR = "airflow_pipeline/mlruns"
CURRENT_MODEL_CONTAINER_PATH = os.path.join(MLRUNS_CONTAINER_DIR, "current_model.json")

DRIFT_BASELINE_PATH = "/opt/airflow/data/drift/baseline_reference.json"


# ─────────────────────────────────────────────
# Helper — pull reviewed images from Postgres
# ─────────────────────────────────────────────
def _load_reviewed_rows(transform, device):
    """Return (images_tensor, labels_array) for all reviewed prediction_logs rows.

    Returns (None, None) if below MIN_REVIEWED_FOR_RETRAIN.
    """
    import io
    import numpy as np
    import psycopg2
    import torch
    from PIL import Image

    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT image_bytes, final_class
            FROM prediction_logs
            WHERE review_status IN ('approved', 'rejected')
              AND final_class IS NOT NULL
            """
        )
        rows = cur.fetchall()

    logging.info(f"Fetched {len(rows)} reviewed rows from predictions-db")

    if len(rows) < MIN_REVIEWED_FOR_RETRAIN:
        logging.warning(
            f"Only {len(rows)} reviewed rows (< {MIN_REVIEWED_FOR_RETRAIN}); "
            "skipping production-data augmentation."
        )
        return None, None

    tensors, labels = [], []
    skipped = 0
    for image_bytes, final_class in rows:
        if final_class not in CLASS_NAME_TO_INDEX:
            skipped += 1
            continue
        try:
            img = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            tensors.append(transform(img))
            labels.append(CLASS_NAME_TO_INDEX[final_class])
        except Exception as e:
            logging.warning(f"Skipping unreadable reviewed row: {e}")
            skipped += 1

    if skipped:
        logging.warning(f"Skipped {skipped} reviewed rows (unknown class or decode error)")

    if not tensors:
        return None, None

    x = torch.stack(tensors).to(device)
    y = np.array(labels, dtype=np.int64)
    logging.info(f"Prepared {len(tensors)} reviewed samples for feature extraction")
    return x, y


def _extract_reviewed_features(model, x_tensor, batch_size=64, use_half=False):
    """Run reviewed images through a backbone; return numpy float32 features."""
    import numpy as np
    import torch

    with torch.inference_mode():
        out_chunks = []
        for i in range(0, len(x_tensor), batch_size):
            chunk = x_tensor[i : i + batch_size]
            if use_half:
                chunk = chunk.half()
            feat = model(chunk).float().cpu().numpy()
            out_chunks.append(feat)
    return np.concatenate(out_chunks, axis=0) if out_chunks else np.empty((0, 0), dtype=np.float32)


# ─────────────────────────────────────────────
# TASK 1 — Feature Extraction
# ─────────────────────────────────────────────
def extract_features_task():
    import gc
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.models import (
        efficientnet_v2_s, EfficientNet_V2_S_Weights,
        resnet34, ResNet34_Weights,
    )
    from datasets import load_dataset

    # ── Reproducibility ──────────────────────
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    torch.set_num_threads(16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = False
    logging.info(f"Device: {device} | half-precision: {use_half}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = "/opt/airflow/data"
    os.makedirs(data_dir, exist_ok=True)

    import datasets
    datasets.utils.logging.set_verbosity_info()
    datasets.utils.logging.enable_progress_bar()

    logging.info("Loading dataset from Hugging Face (this may take 2-5 min)...")
    dataset = load_dataset("Punnarunwuwu/seml-industry-ver")
    logging.info("Dataset loaded. Separating splits...")
    train_v1_dataset = dataset["train_v1"]
    train_v2_dataset = dataset["train_v2"]
    test_dataset  = dataset["test"]
    logging.info(
        f"Train_v1 size: {len(train_v1_dataset)} | "
        f"Train_v2 size: {len(train_v2_dataset)} | "
        f"Test size: {len(test_dataset)}"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def apply_transforms(batch):
        pixel_values = []
        for img in batch["image"]:
            try:
                pixel_values.append(transform(img.convert("RGB")))
            except Exception:
                pixel_values.append(torch.zeros(3, 224, 224))
        batch["pixel_values"] = pixel_values
        return batch

    train_v1_dataset.set_transform(apply_transforms)
    train_v2_dataset.set_transform(apply_transforms)
    test_dataset.set_transform(apply_transforms)

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels       = torch.tensor([item["label"]        for item in batch])
        return pixel_values, labels

    LOADER_KWARGS = dict(
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    train_v1_loader = DataLoader(train_v1_dataset, **LOADER_KWARGS)
    train_v2_loader = DataLoader(train_v2_dataset, **LOADER_KWARGS)
    test_loader  = DataLoader(test_dataset,  **LOADER_KWARGS)

    # Reviewed production data (same transform as HF dataset above).
    reviewed_x, reviewed_y = _load_reviewed_rows(transform, device)

    @torch.inference_mode()
    def extract_and_save(loader, split_name: str, model: nn.Module,
                         out_path: str):
        total_batches = len(loader)

        sample_x, _ = next(iter(loader))
        sample_x = sample_x[:1].to(device)
        if use_half:
            sample_x = sample_x.half()
        with torch.inference_mode():
            sample_out = model(sample_x)
        feat_dim = sample_out.shape[1]
        del sample_out

        n_samples = len(loader.dataset)
        mmap = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32,
            shape=(n_samples, feat_dim),
        )

        idx = 0
        for i, (x, _) in enumerate(loader):
            if i % 20 == 0 or i == total_batches - 1:
                logging.info(f"[{split_name}] batch {i+1}/{total_batches}")

            x = x.to(device, non_blocking=True)
            if use_half:
                x = x.half()

            out = model(x).float().cpu().numpy()
            bsz = out.shape[0]
            mmap[idx: idx + bsz] = out
            idx += bsz

            del x, out

        mmap.flush()
        del mmap
        logging.info(f"Saved {split_name} features → {out_path}")

    # ────────────────────────────────────────────
    # MODEL 1 — ResNet34
    # ────────────────────────────────────────────
    logging.info("Loading ResNet34 backbone...")
    backbone = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
    if use_half:
        backbone = backbone.half()
    backbone.fc = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    extract_and_save(train_v1_loader, "TRAIN_V1/ResNet34", backbone,
                     os.path.join(data_dir, "X_train_v1_resnet.npy"))
    extract_and_save(train_v2_loader, "TRAIN_V2/ResNet34", backbone,
                     os.path.join(data_dir, "X_train_v2_resnet.npy"))
    extract_and_save(test_loader,  "TEST/ResNet34",  backbone,
                     os.path.join(data_dir, "X_test_resnet.npy"))

    # Reviewed features via same ResNet34 backbone
    if reviewed_x is not None:
        reviewed_feats = _extract_reviewed_features(backbone, reviewed_x, use_half=use_half)
        np.save(os.path.join(data_dir, "X_reviewed_resnet.npy"), reviewed_feats)
        np.save(os.path.join(data_dir, "y_reviewed.npy"), reviewed_y)
        logging.info(f"Saved reviewed ResNet34 features: {reviewed_feats.shape}")
    else:
        # Clear any stale reviewed features from earlier runs
        for stale in ("X_reviewed_resnet.npy", "X_reviewed_effnet.npy", "y_reviewed.npy"):
            p = os.path.join(data_dir, stale)
            if os.path.exists(p):
                os.remove(p)

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save labels ──
    logging.info("Saving labels...")
    y_train_v1_list, y_train_v2_list, y_test_list = [], [], []
    for _, y in train_v1_loader:
        y_train_v1_list.append(y.numpy())
    for _, y in train_v2_loader:
        y_train_v2_list.append(y.numpy())
    for _, y in test_loader:
        y_test_list.append(y.numpy())
    np.save(os.path.join(data_dir, "y_train_v1.npy"), np.concatenate(y_train_v1_list))
    np.save(os.path.join(data_dir, "y_train_v2.npy"), np.concatenate(y_train_v2_list))
    np.save(os.path.join(data_dir, "y_test.npy"),  np.concatenate(y_test_list))
    del y_train_v1_list, y_train_v2_list, y_test_list
    gc.collect()

    # ────────────────────────────────────────────
    # MODEL 2 — EfficientNetV2-S
    # ────────────────────────────────────────────
    logging.info("Loading EfficientNetV2-S backbone...")
    backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
    if use_half:
        backbone = backbone.half()
    backbone.classifier = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    extract_and_save(train_v1_loader, "TRAIN_V1/EffNetV2-S", backbone,
                     os.path.join(data_dir, "X_train_v1_effnet.npy"))
    extract_and_save(train_v2_loader, "TRAIN_V2/EffNetV2-S", backbone,
                     os.path.join(data_dir, "X_train_v2_effnet.npy"))
    extract_and_save(test_loader,  "TEST/EffNetV2-S",  backbone,
                     os.path.join(data_dir, "X_test_effnet.npy"))

    if reviewed_x is not None:
        reviewed_feats = _extract_reviewed_features(backbone, reviewed_x, use_half=use_half)
        np.save(os.path.join(data_dir, "X_reviewed_effnet.npy"), reviewed_feats)
        logging.info(f"Saved reviewed EfficientNetV2-S features: {reviewed_feats.shape}")

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"All features saved to {data_dir}")


# ─────────────────────────────────────────────
# TASK 2 — Train & Log
# ─────────────────────────────────────────────
def _augment_with_reviewed(X_train, y_train, data_dir, reviewed_feat_file):
    """Concatenate reviewed features + labels to the HF-derived arrays."""
    import numpy as np

    reviewed_feat_path = os.path.join(data_dir, reviewed_feat_file)
    reviewed_y_path = os.path.join(data_dir, "y_reviewed.npy")

    if not (os.path.exists(reviewed_feat_path) and os.path.exists(reviewed_y_path)):
        return X_train, y_train, 0

    reviewed_X = np.load(reviewed_feat_path)
    reviewed_y = np.load(reviewed_y_path)

    if reviewed_X.shape[0] == 0:
        return X_train, y_train, 0

    combined_X = np.concatenate([X_train, reviewed_X], axis=0)
    combined_y = np.concatenate([y_train, reviewed_y], axis=0)
    return combined_X, combined_y, int(reviewed_X.shape[0])


def _resolve_pkl_path(model_info) -> str | None:
    """Return a HOST-relative path to the pickled model for current_model.json."""
    import mlflow

    # MLflow 3: ModelInfo has `.model_id` = 'm-<uuid>'. Layout is
    #   {artifact_root}/{experiment_id}/models/{model_id}/artifacts/model.pkl
    model_id = getattr(model_info, "model_id", None)
    exp_id = None
    try:
        run = mlflow.get_run(model_info.run_id)
        exp_id = run.info.experiment_id
    except Exception:
        pass

    if model_id and exp_id:
        return f"{MLRUNS_HOST_DIR}/{exp_id}/models/{model_id}/artifacts/model.pkl"

    # Fallback: derive from artifact_uri (MLflow 2 legacy layout).
    try:
        artifact_uri = model_info.artifact_uri  # e.g. file:///.../mlruns/1/<run>/artifacts/pca_model
        if artifact_uri and "/opt/airflow/mlruns/" in artifact_uri:
            container_path = artifact_uri.split("file://")[-1]
            host_relative = container_path.replace("/opt/airflow/mlruns", MLRUNS_HOST_DIR, 1)
            return f"{host_relative}/model.pkl"
    except Exception:
        pass

    return None


def train_and_log_task():
    import subprocess
    import sys
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
        import matplotlib.pyplot as plt
        import seaborn as sns

    import json
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.decomposition import PCA

    data_dir = "/opt/airflow/data"
    y_train_v1 = np.load(os.path.join(data_dir, "y_train_v1.npy"))
    y_train_v2 = np.load(os.path.join(data_dir, "y_train_v2.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    logging.info(f"MLflow URI: {MLFLOW_URI}")
    mlflow.set_experiment("Places365_Classification")

    models_to_train = [
        {"name": "ResNet34",         "ds_name": "train_v1", "train_file": "X_train_v1_resnet.npy", "test_file": "X_test_resnet.npy", "reviewed_feat_file": "X_reviewed_resnet.npy", "y_train": y_train_v1},
        {"name": "ResNet34",         "ds_name": "train_v2", "train_file": "X_train_v2_resnet.npy", "test_file": "X_test_resnet.npy", "reviewed_feat_file": "X_reviewed_resnet.npy", "y_train": y_train_v2},
        {"name": "EfficientNetV2-S", "ds_name": "train_v1", "train_file": "X_train_v1_effnet.npy", "test_file": "X_test_effnet.npy", "reviewed_feat_file": "X_reviewed_effnet.npy", "y_train": y_train_v1},
        {"name": "EfficientNetV2-S", "ds_name": "train_v2", "train_file": "X_train_v2_effnet.npy", "test_file": "X_test_effnet.npy", "reviewed_feat_file": "X_reviewed_effnet.npy", "y_train": y_train_v2},
    ]

    best_f1 = 0.0
    best_record = None  # winning config's pca + clf ModelInfo
    run_results = []

    for cfg in models_to_train:
        X_train = np.load(os.path.join(data_dir, cfg["train_file"]))
        X_test  = np.load(os.path.join(data_dir, cfg["test_file"]))
        y_train = cfg["y_train"]

        X_train, y_train, reviewed_added = _augment_with_reviewed(
            X_train, y_train, data_dir, cfg["reviewed_feat_file"],
        )

        with mlflow.start_run(run_name=f"PCA256_LogReg_{cfg['name']}_{cfg['ds_name']}") as run:
            components = 256
            mlflow.log_params({
                "pca_components": components,
                "classifier":     "LogisticRegression",
                "backbone":       cfg["name"],
                "dataset_split":  cfg["ds_name"],
                "dataset":        "Punnarunwuwu/seml-industry-ver",
                "reviewed_samples_added": reviewed_added,
            })

            pca = PCA(n_components=components, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca  = pca.transform(X_test)
            del X_train, X_test

            clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
            clf.fit(X_train_pca, y_train)

            train_preds = clf.predict(X_train_pca)
            test_preds  = clf.predict(X_test_pca)

            train_acc = accuracy_score(y_train, train_preds)
            train_f1  = f1_score(y_train, train_preds, average="macro")
            test_acc  = accuracy_score(y_test, test_preds)
            test_f1   = f1_score(y_test,  test_preds,  average="macro")

            logging.info(
                f"[{cfg['name']} | {cfg['ds_name']}] TRAIN acc={train_acc:.4f} f1={train_f1:.4f} "
                f"(+{reviewed_added} reviewed samples)"
            )
            logging.info(
                f"[{cfg['name']} | {cfg['ds_name']}] TEST  acc={test_acc:.4f}  f1={test_f1:.4f}"
            )

            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "train_f1":       float(train_f1),
                "test_accuracy":  test_acc,
                "test_f1":        float(test_f1),
            })
            pca_info = mlflow.sklearn.log_model(pca, "pca_model")
            clf_info = mlflow.sklearn.log_model(clf, "logistic_regression_model")

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_record = {
                    "pca_info": pca_info,
                    "clf_info": clf_info,
                    "run_id": run.info.run_id,
                    "config_name": f"{cfg['name']} / {cfg['ds_name']}",
                    "test_f1": float(test_f1),
                }

            run_results.append({
                "name": f"{cfg['name']}\n({cfg['ds_name']})",
                "f1": test_f1
            })

    if best_record is None:
        logging.error("No successful training runs — skipping model promotion.")
    else:
        pca_pkl_path = _resolve_pkl_path(best_record["pca_info"])
        clf_pkl_path = _resolve_pkl_path(best_record["clf_info"])
        if pca_pkl_path and clf_pkl_path:
            pointer = {
                "pca_run_id": best_record["run_id"],
                "clf_run_id": best_record["run_id"],
                "pca_pkl_path": pca_pkl_path,
                "clf_pkl_path": clf_pkl_path,
                "winning_config": best_record["config_name"],
                "test_f1": best_record["test_f1"],
                "promoted_at": datetime.now().isoformat() + "Z",
            }
            os.makedirs(os.path.dirname(CURRENT_MODEL_CONTAINER_PATH), exist_ok=True)
            with open(CURRENT_MODEL_CONTAINER_PATH, "w", encoding="utf-8") as f:
                json.dump(pointer, f, indent=2, ensure_ascii=False)
            logging.info(
                f"Promoted {best_record['config_name']} (F1={best_f1:.4f}) → {CURRENT_MODEL_CONTAINER_PATH}"
            )
        else:
            logging.error(
                "Could not resolve artifact paths from ModelInfo; skipping promotion. "
                f"pca_info={best_record['pca_info']}, clf_info={best_record['clf_info']}"
            )

    logging.info(
        f"Training complete. Best: {best_record['config_name'] if best_record else 'n/a'} — F1={best_f1:.4f}"
    )

    try:
        plt.figure(figsize=(12, 6))
        names = [r["name"] for r in run_results]
        f1_scores = [r["f1"] for r in run_results]
        sns.barplot(x=names, y=f1_scores, palette="viridis")
        plt.title("Model F1 Score Comparison Across Datasets")
        plt.ylabel("Test F1 Score")
        plt.ylim(0, 1.0)

        out_path = os.path.join(data_dir, "f1_comparison_chart.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        with mlflow.start_run(run_name="Visualization_Report"):
            mlflow.log_artifact(out_path)
            if best_record:
                mlflow.log_param("winning_model", best_record["config_name"])
            mlflow.log_metric("winning_f1", best_f1)

    except Exception as e:
        logging.error(f"Failed to generate MLflow visualization: {e}")


# ─────────────────────────────────────────────
# TASK 3 — Regenerate drift baseline
# ─────────────────────────────────────────────
def regenerate_baseline_task():
    if os.path.exists(DRIFT_BASELINE_PATH):
        os.remove(DRIFT_BASELINE_PATH)
        logging.info(f"Deleted {DRIFT_BASELINE_PATH}; drift DAG will re-bootstrap.")
    else:
        logging.info(f"No baseline file at {DRIFT_BASELINE_PATH}; nothing to regenerate.")


# ─────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────
default_args = {
    "owner":            "ml_engineer",
    "depends_on_past":  False,
    "start_date":       datetime(2023, 1, 1),
    "retries":          0,
    "retry_delay":      timedelta(minutes=5),
}

with DAG(
    "model_training_pipeline",
    default_args=default_args,
    description="Training pipeline comparing ResNet34 and EfficientNetV2-S",
    schedule=None,
    catchup=False,
    tags=["image_classification", "mlflow"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features_task,
    )
    train_log_task = PythonOperator(
        task_id="train_and_evaluate_models",
        python_callable=train_and_log_task,
    )
    regenerate_baseline = PythonOperator(
        task_id="regenerate_baseline",
        python_callable=regenerate_baseline_task,
    )

    extract_task >> train_log_task >> regenerate_baseline
