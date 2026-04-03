import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

def extract_features_task():
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    from datasets import load_dataset
    from tqdm import tqdm

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading dataset from Hugging Face...")
    dataset = load_dataset("Punnarunwuwu/industry-verification-seml")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    dataset.set_transform(apply_transforms)

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["validation"], batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)

    logging.info("Loading EfficientNetV2-S backbone...")
    backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
    backbone.classifier = nn.Identity()

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.eval()

    @torch.no_grad()
    def extract_features(loader, name):
        feats, labels = [], []
        # Using simple iteration instead of tqdm to keep logs clean in Airflow
        for x, y in loader:
            x = x.to(device)
            output = backbone(x)
            feats.append(output.cpu().numpy())
            labels.append(y.numpy())
        return np.vstack(feats), np.hstack(labels)

    logging.info("Running Feature Extraction for TRAIN...")
    X_train, y_train = extract_features(train_loader, "TRAIN")
    logging.info("Running Feature Extraction for TEST...")
    X_test, y_test = extract_features(test_loader, "TEST")

    # Saving data locally to be picked up by the next task
    data_dir = "/opt/airflow/data"
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    logging.info(f"Saved extracted features to {data_dir}")
    logging.info(f"Train features shape: {X_train.shape}")
    logging.info(f"Test features shape: {X_test.shape}")


def train_and_log_task():
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.decomposition import PCA

    data_dir = "/opt/airflow/data"
    
    logging.info("Loading extracted features...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    logging.info(f"Using MLflow tracking URI: {MLFLOW_URI}")
    
    experiment_name = "Places365_EfficientNetV2S"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="PCA256_LogisticRegression"):
        # Log parameters
        components = 256
        mlflow.log_param("pca_components", components)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("backbone", "EfficientNetV2-S")
        mlflow.log_param("dataset", "Punnarunwuwu/industry-verification-seml")

        logging.info("Applying PCA (Reducing to 256 dimensions)...")
        pca = PCA(n_components=components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        logging.info("Training Logistic Regression...")
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
        clf.fit(X_train_pca, y_train)

        # Evaluate
        train_preds = clf.predict(X_train_pca)
        test_preds = clf.predict(X_test_pca)

        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average="macro")
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro")

        logging.info(f"TRAIN | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
        logging.info(f"TEST  | Accuracy: {test_acc:.4f}  | F1: {test_f1:.4f}")

        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "train_f1": float(train_f1),
            "test_accuracy": test_acc,
            "test_f1": float(test_f1)
        })

        # Log models
        logging.info("Logging models to MLflow...")
        mlflow.sklearn.log_model(pca, "pca_model")
        mlflow.sklearn.log_model(clf, "logistic_regression_model")
        
        logging.info("Training and Logging completed successfully.")


default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'efficientnet_training_pipeline',
    default_args=default_args,
    description='A simple training pipeline with EfficientNetV2-S, PCA, and Logistic Regression',
    schedule_interval=None,
    catchup=False,
    tags=['image_classification', 'mlflow']
) as dag:

    # 1. Extract Features Task
    extract_task = PythonOperator(
        task_id='extract_features_efficientnet',
        python_callable=extract_features_task,
    )

    # 2. Train & Log Task
    train_log_task = PythonOperator(
        task_id='train_and_log_mlflow',
        python_callable=train_and_log_task,
    )

    # Define dependencies
    extract_task >> train_log_task
