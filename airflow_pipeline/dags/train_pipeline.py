import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging


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
    from datasets import load_dataset, concatenate_datasets

    # ── Reproducibility ──────────────────────
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Thread budget — harness huge RAM and CPU cores ──
    torch.set_num_threads(16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Quadro P4000 (SM 6.1) lacks hardware FP16 cores, so we must strictly use FP32.
    use_half = False
    logging.info(f"Device: {device} | half-precision: {use_half}")

    # ── Authenticated fast HF download ───────
    # os.environ["HF_TOKEN"] = ""

    # hf_transfer: Rust-based downloader, up to 5× faster.
    # Requires: pip install hf_transfer  (safe no-op if not installed)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # TOKENIZERS_PARALLELISM=false + DATASETS_VERBOSITY=debug help avoid
    # deadlocks caused by forked processes inside the Airflow worker.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Disable the datasets progress bar — it can block in non-TTY envs
    # os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

    data_dir = "/opt/airflow/data"
    os.makedirs(data_dir, exist_ok=True)

    # ── Load dataset ─────────────────────────
    # IMPORTANT: num_proc is intentionally omitted here.
    # Inside an Airflow worker (which is itself a forked process),
    # spawning child processes via num_proc causes a deadlock / hang
    # because the multiprocessing start method defaults to "fork" and
    # the HF datasets library uses semaphores that don't survive a fork.
    # Single-process download is slower per-shard but never hangs.
    import datasets
    datasets.utils.logging.set_verbosity_info()
    datasets.utils.logging.enable_progress_bar()
    
    logging.info("Loading dataset from Hugging Face (this may take 2-5 min)...")
    dataset = load_dataset("Punnarunwuwu/seml-industry-ver")
    logging.info("Dataset loaded. Separating splits...")
    train_v1_dataset = dataset["train_v1"]
    train_v2_dataset = dataset["train_v2"]
    test_dataset  = dataset["test"]
    logging.info(f"Train_v1 size: {len(train_v1_dataset)} | Train_v2 size: {len(train_v2_dataset)} | Test size: {len(test_dataset)}")

    # ── Transforms ───────────────────────────
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

    # num_workers=0 — mandatory inside Airflow.
    # Airflow workers are themselves forked processes; spawning further
    # child workers causes deadlocks with PyTorch's multiprocessing.
    # The GPU keeps utilisation high enough that the single-process
    # loader is not the bottleneck here — inference dominates.
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

    # ── Shared extraction helper ──────────────
    @torch.inference_mode()
    def extract_and_save(loader, split_name: str, model: nn.Module,
                         out_path: str):
        """Extract features into a pre-allocated memmap to avoid holding
        the entire array in RAM at once."""
        total_batches = len(loader)

        # First pass: figure out feature dim from one batch
        sample_x, _ = next(iter(loader))
        sample_x = sample_x[:1].to(device)
        if use_half:
            sample_x = sample_x.half()
        with torch.inference_mode():
            sample_out = model(sample_x)
        feat_dim = sample_out.shape[1]
        del sample_out

        n_samples = len(loader.dataset)
        # memmap writes directly to disk — never loads the full array
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

            out = model(x).float().cpu().numpy()   # back to float32 for saving
            bsz = out.shape[0]
            mmap[idx: idx + bsz] = out
            idx += bsz

            # Explicit per-batch cleanup — keeps peak RAM flat
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

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save labels (only need to do this once, from ResNet pass) ──
    # Re-collect labels in a single pass without model inference
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

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"All features saved to {data_dir}")


# ─────────────────────────────────────────────
# TASK 2 — Train & Log
# ─────────────────────────────────────────────
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
        {"name": "ResNet34", "ds_name": "train_v1", "train_file": "X_train_v1_resnet.npy", "test_file": "X_test_resnet.npy", "y_train": y_train_v1},
        {"name": "ResNet34", "ds_name": "train_v2", "train_file": "X_train_v2_resnet.npy", "test_file": "X_test_resnet.npy", "y_train": y_train_v2},
        {"name": "EfficientNetV2-S", "ds_name": "train_v1", "train_file": "X_train_v1_effnet.npy", "test_file": "X_test_effnet.npy", "y_train": y_train_v1},
        {"name": "EfficientNetV2-S", "ds_name": "train_v2", "train_file": "X_train_v2_effnet.npy", "test_file": "X_test_effnet.npy", "y_train": y_train_v2},
    ]

    best_f1 = 0.0
    best_config_name = ""
    run_results = []

    for cfg in models_to_train:
        # Huge RAM is available, load arrays fully into memory instead of streaming disk
        X_train = np.load(os.path.join(data_dir, cfg["train_file"]))
        X_test  = np.load(os.path.join(data_dir, cfg["test_file"]))
        y_train = cfg["y_train"]

        with mlflow.start_run(run_name=f"PCA256_LogReg_{cfg['name']}_{cfg['ds_name']}"):
            components = 256
            mlflow.log_params({
                "pca_components": components,
                "classifier":     "LogisticRegression",
                "backbone":       cfg["name"],
                "dataset_split":  cfg["ds_name"],
                "dataset":        "Punnarunwuwu/seml-industry-ver",
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
                f"[{cfg['name']} | {cfg['ds_name']}] TRAIN acc={train_acc:.4f} f1={train_f1:.4f}"
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
            mlflow.sklearn.log_model(pca, "pca_model")
            mlflow.sklearn.log_model(clf, "logistic_regression_model")

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_config_name = f"{cfg['name']} trained on {cfg['ds_name']}"

            run_results.append({
                "name": f"{cfg['name']}\n({cfg['ds_name']})",
                "f1": test_f1
            })

    logging.info(
        f"Training complete. Best: {best_config_name} — F1={best_f1:.4f}"
    )

    try:
        plt.figure(figsize=(12, 6))
        names = [r["name"] for r in run_results]
        f1_scores = [r["f1"] for r in run_results]
        sns.barplot(x=names, y=f1_scores, palette="viridis")
        plt.title("Model F1 Score Comparison Across Datasets")
        plt.ylabel("Test F1 Score")
        plt.ylim(0, 1.0)
        
        # Determine data_dir if it's not strictly absolute path globally
        out_path = os.path.join(data_dir, "f1_comparison_chart.png") if 'data_dir' in locals() else "/tmp/f1_comparison_chart.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        with mlflow.start_run(run_name="Visualization_Report"):
            mlflow.log_artifact(out_path)
            mlflow.log_param("winning_model", best_config_name.replace('\n', ' '))
            mlflow.log_metric("winning_f1", best_f1)
            
    except Exception as e:
        logging.error(f"Failed to generate MLflow visualization: {e}")


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

    extract_task >> train_log_task