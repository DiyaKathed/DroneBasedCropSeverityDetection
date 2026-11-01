"""
Evaluation script for the PlantDoc Disease Classification Model.
Computes overall accuracy, macro precision/recall/F1, weighted F1,
per-class metrics, and confusion matrix on the held-out test set.
"""

import json
import os
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Paths
BASE_DIR = Path("Data/PlantDoc-Dataset-master/test")
MODEL_PATH = "models/plantdoc_disease_model.keras"
LABELS_PATH = "class_labels.json"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(model_path=MODEL_PATH, labels_path=LABELS_PATH, test_dir=BASE_DIR):
    print("=" * 60)
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, "r") as f:
        class_labels = json.load(f)
    if isinstance(class_labels, dict):
        class_labels = [class_labels[str(i)] for i in range(len(class_labels))]
        
    print(f"Loaded {len(class_labels)} class labels.")
    print(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
    
    # Collect test images and true labels
    image_paths = []
    true_labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    missing_classes = []
    for c_idx, c_name in enumerate(class_labels):
        c_folder = test_dir / c_name
        if not c_folder.exists() or not c_folder.is_dir():
            missing_classes.append(c_name)
            continue
        for img_path in sorted(c_folder.iterdir()):
            if img_path.suffix.lower() in valid_extensions and not img_path.name.startswith("."):
                image_paths.append(img_path)
                true_labels.append(c_idx)

    print(f"Total test images found: {len(image_paths)}")
    if missing_classes:
        print(f"Note: {len(missing_classes)} classes not present in test folder: {missing_classes}")

    # Load images in batches as raw RGB [0, 255] float32
    # The models contain internal Rescaling layers matching their training configuration.
    batch_images = []
    valid_true = []
    
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB").resize((224, 224))
                arr = np.array(img_rgb, dtype=np.float32)
                batch_images.append(arr)
                valid_true.append(true_labels[i])
        except Exception as e:
            print(f"Warning: could not process {path}: {e}")

    X_test = np.array(batch_images, dtype=np.float32)
    y_true = np.array(valid_true)

    print(f"Running predictions on {len(X_test)} images...")
    predictions = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test Images Evaluated: {len(y_true)}")
    print(f"Accuracy:             {acc * 100:.2f}%")
    print(f"Macro Precision:      {macro_p * 100:.2f}%")
    print(f"Macro Recall:         {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:       {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:    {weighted_f1 * 100:.2f}%")

    present_indices = sorted(list(set(y_true)))
    present_names = [class_labels[i] for i in present_indices]

    print("\nPer-Class Classification Report:")
    print("-" * 60)
    rep = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=present_names,
        zero_division=0,
    )
    print(rep)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    # Save metrics JSON
    metrics = {
        "model_path": str(model_path),
        "test_images": int(len(y_true)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    metrics_save_path = REPORT_DIR / "test_evaluation_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    return metrics, cm

if __name__ == "__main__":
    evaluate()