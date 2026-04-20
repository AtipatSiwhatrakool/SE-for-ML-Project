import numpy as np
import pickle
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Paths
PCA_PATH = "airflow_pipeline/mlruns/1/models/m-99ea752fc1b7401e997dfe755d74e8d6/artifacts/model.pkl"
CLF_PATH = "airflow_pipeline/mlruns/1/models/m-1b25cd047c0d48f3abda57a99fdf9f38/artifacts/model.pkl"
IMAGE_PATH = "salon.jpg"

CLASS_NAMES = {
    0: "beauty_salon",
    1: "drugstore",
    2: "restaurant",
    3: "movie_theater",
    4: "apartment_building",
    5: "supermarket",
}

# Load models
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)

with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

print(f"PCA input features: {pca.n_features_in_}")
print(f"PCA output features: {pca.n_components_}")
print(f"Classifier classes: {clf.classes_}")

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"\nImage loaded: {image.size}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img_tensor = transform(image).unsqueeze(0)
print(f"Input tensor shape: {img_tensor.shape}")

# Extract features with EfficientNetV2-S
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
backbone.classifier = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

with torch.inference_mode():
    features = backbone(img_tensor.to(device)).cpu().numpy()

print(f"Feature shape: {features.shape} (should be 1280)")

# Apply PCA
features_pca = pca.transform(features)
print(f"PCA features shape: {features_pca.shape}")

# Predict
pred = clf.predict(features_pca)[0]
proba = clf.predict_proba(features_pca)[0]

print(f"\n{'='*40}")
print(f"Predicted class: {pred} ({CLASS_NAMES[pred]})")
print(f"Confidence: {proba[pred]:.4f}")
print(f"\nAll probabilities:")
for i, p in enumerate(proba):
    print(f"  {i} ({CLASS_NAMES[i]}): {p:.4f}")
print(f"{'='*40}")
