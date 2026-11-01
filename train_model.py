"""
Reproducible Training Pipeline for Drone-Based Crop Disease Classification.
Compares MobileNetV2 and EfficientNetB0 on the PlantDoc dataset.
Features:
 - Data leakage prevention (deduplication against test set)
 - Stratified train/validation split
 - Data augmentation preserving disease features
 - Class weight computation for extreme class imbalance
 - Two-stage transfer learning (frozen feature extractor -> gentle fine-tuning)
 - Early stopping & best checkpoint recovery
 - Comprehensive metric tracking (Accuracy, Macro/Weighted F1, Loss)
"""

import os
import ssl
import json
import hashlib
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Fix SSL context for model weight downloads on macOS
try:
    # pyrefly: ignore [missing-import]
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = Path("Data/PlantDoc-Dataset-master/train")
TEST_DIR = Path("Data/PlantDoc-Dataset-master/test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def prepare_dataset():
    """
    Loads dataset, dedupes against test set, builds class mapping,
    and returns stratified train/val splits.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA INGESTION & LEAKAGE AUDIT")
    print("=" * 60)

    # Hashes of test images to avoid train/test contamination
    test_hashes = set()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in TEST_DIR.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
            test_hashes.add(get_file_hash(p))
    print(f"Collected {len(test_hashes)} unique test image hashes.")

    class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Identified {len(class_names)} disease classes in train directory.")

    # Save canonical class mapping
    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=4)
    print("Saved canonical class mapping to class_labels.json.")

    image_paths = []
    labels = []
    leakage_count = 0

    for idx, c_name in enumerate(class_names):
        c_dir = TRAIN_DIR / c_name
        for img_path in sorted(c_dir.iterdir()):
            if img_path.is_file() and not img_path.name.startswith(".") and img_path.suffix.lower() in valid_exts:
                h = get_file_hash(img_path)
                if h in test_hashes:
                    leakage_count += 1
                    # Skip duplicate from training to preserve untouched test integrity
                    continue
                image_paths.append(str(img_path))
                labels.append(idx)

    print(f"Filtered out {leakage_count} images that duplicated test set samples.")
    print(f"Total clean training pool: {len(image_paths)} images across {len(class_names)} classes.")

    # Stratified split: 80% train, 20% validation
    # For classes with only 1 sample after deduplication, handle gracefully
    counts = np.bincount(labels, minlength=len(class_names))
    rare_classes = [i for i, c in enumerate(counts) if c < 2]
    if rare_classes:
        print(f"Warning: Classes with < 2 samples: {[class_names[i] for i in rare_classes]}")

    train_paths, val_paths, train_y, val_y = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels if len(rare_classes) == 0 else None
    )

    print(f"Train split: {len(train_paths)} images")
    print(f"Val split:   {len(val_paths)} images")

    # Compute balanced class weights to address severe class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_y),
        y=np.array(train_y)
    )
    # Clip weights to avoid gradient spikes on tiny classes
    class_weights = np.clip(class_weights, 0.2, 10.0)
    weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(train_y), class_weights)}

    return train_paths, train_y, val_paths, val_y, class_names, weight_dict

def load_and_preprocess_image(path, label, num_classes):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    # Convert to float32 [0, 255]. Rescaling layers inside model handle normalization.
    img = tf.cast(img, tf.float32)
    label_one_hot = tf.one_hot(label, num_classes)
    return img, label_one_hot

def create_tf_dataset(paths, labels, num_classes, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(
        lambda p, l: load_and_preprocess_image(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_data_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.12, fill_mode="reflect", seed=SEED),
        layers.RandomZoom(0.12, fill_mode="reflect", seed=SEED),
        layers.RandomContrast(0.12, seed=SEED),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=SEED),
    ], name="data_augmentation")

def build_model(architecture_name, num_classes):
    """
    Builds a transfer learning model with integrated data augmentation and internal rescaling.
    """
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    aug = build_data_augmentation()(inputs)

    if architecture_name == "MobileNetV2":
        # MobileNetV2 expects [-1, 1] inputs:
        rescaling = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(aug)
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=rescaling
        )
        base_model.trainable = False
        features = base_model.output

    elif architecture_name == "EfficientNetB0":
        # EfficientNetB0 has its own normalization internally, expects [0, 255] RGB:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=aug
        )
        base_model.trainable = False
        features = base_model.output
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="features_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"PlantDoc_{architecture_name}")
    return model, base_model

def train_and_finetune(
    architecture_name,
    train_ds,
    val_ds,
    num_classes,
    weight_dict,
    stage1_epochs=6,
    stage2_epochs=6
):
    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {architecture_name}")
    print("=" * 60)

    model, base_model = build_model(architecture_name, num_classes)

    # -----------------------------
    # Stage 1: Train Head Only
    # -----------------------------
    print(f"\n--- Stage 1: Feature Extraction (Top layers only, {stage1_epochs} epochs) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = MODELS_DIR / f"{architecture_name.lower()}_best_val.keras"
    callbacks_stage1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True)
    ]

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=weight_dict,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # -----------------------------
    # Stage 2: Gentle Fine-Tuning
    # -----------------------------
    print(f"\n--- Stage 2: Fine-Tuning Base Layers ({stage2_epochs} epochs) ---")
    base_model.trainable = True
    
    # Freeze lower layers, unfreeze top block
    if architecture_name == "MobileNetV2":
        for layer in base_model.layers[:-35]:
            layer.trainable = False
    elif architecture_name == "EfficientNetB0":
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True)
    ]

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=weight_dict,
        callbacks=callbacks_stage2,
        verbose=1
    )

    # Combine histories
    combined_hist = {}
    for k in h1.history.keys():
        combined_hist[k] = h1.history[k] + h2.history[k]

    # Save plot of learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(combined_hist["accuracy"], label="Train Acc")
    ax1.plot(combined_hist["val_accuracy"], label="Val Acc")
    ax1.axvline(x=len(h1.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="Fine-tune Start")
    ax1.set_title(f"{architecture_name} Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(combined_hist["loss"], label="Train Loss")
    ax2.plot(combined_hist["val_loss"], label="Val Loss")
    ax2.axvline(x=len(h1.history["loss"]) - 0.5, color="gray", linestyle="--", label="Fine-tune Start")
    ax2.set_title(f"{architecture_name} Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plot_path = REPORTS_DIR / f"{architecture_name.lower()}_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Load best checkpoint for evaluation
    best_model = keras.models.load_model(str(checkpoint_path))
    return best_model, combined_hist, checkpoint_path

def evaluate_on_val(model, val_paths, val_y, class_names):
    """
    Evaluates model on validation split.
    """
    images = []
    for p in val_paths:
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
            images.append(arr)
    X_val = np.array(images)
    preds = model.predict(X_val, batch_size=32, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(val_y, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(val_y, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(val_y, y_pred, average="weighted", zero_division=0)

    return {
        "val_accuracy": float(acc),
        "val_macro_f1": float(macro_f1),
        "val_weighted_f1": float(weighted_f1)
    }

def main():
    train_paths, train_y, val_paths, val_y, class_names, weight_dict = prepare_dataset()
    num_classes = len(class_names)

    train_ds = create_tf_dataset(train_paths, train_y, num_classes, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_y, num_classes, is_training=False)

    # 1. Train MobileNetV2 (5 epochs stage 1 + 5 epochs stage 2)
    mobilenet_model, mobilenet_hist, mobilenet_ckpt = train_and_finetune(
        "MobileNetV2",
        train_ds,
        val_ds,
        num_classes,
        weight_dict,
        stage1_epochs=5,
        stage2_epochs=5
    )
    mobilenet_val = evaluate_on_val(mobilenet_model, val_paths, val_y, class_names)
    print(f"MobileNetV2 Val Results: Acc = {mobilenet_val['val_accuracy']*100:.2f}%, Macro F1 = {mobilenet_val['val_macro_f1']*100:.2f}%")

    # 2. Train EfficientNetB0 (5 epochs stage 1 + 5 epochs stage 2)
    effnet_model, effnet_hist, effnet_ckpt = train_and_finetune(
        "EfficientNetB0",
        train_ds,
        val_ds,
        num_classes,
        weight_dict,
        stage1_epochs=5,
        stage2_epochs=5
    )
    effnet_val = evaluate_on_val(effnet_model, val_paths, val_y, class_names)
    print(f"EfficientNetB0 Val Results: Acc = {effnet_val['val_accuracy']*100:.2f}%, Macro F1 = {effnet_val['val_macro_f1']*100:.2f}%")

    # 3. Model Comparison & Selection
    print("\n" + "=" * 60)
    print("STEP 2: MODEL COMPARISON & SELECTION")
    print("=" * 60)
    print(f"MobileNetV2:    Val Accuracy = {mobilenet_val['val_accuracy']*100:.2f}%, Val Macro F1 = {mobilenet_val['val_macro_f1']*100:.2f}%")
    print(f"EfficientNetB0: Val Accuracy = {effnet_val['val_accuracy']*100:.2f}%, Val Macro F1 = {effnet_val['val_macro_f1']*100:.2f}%")

    if effnet_val["val_accuracy"] >= mobilenet_val["val_accuracy"]:
        selected_name = "EfficientNetB0"
        selected_model = effnet_model
        selected_ckpt = effnet_ckpt
    else:
        selected_name = "MobileNetV2"
        selected_model = mobilenet_model
        selected_ckpt = mobilenet_ckpt

    print(f"\nChampion Model Selected: {selected_name}!")
    
    # Save the selected champion model to standard deployment location
    deploy_path = MODELS_DIR / "plantdoc_disease_model.keras"
    selected_model.save(str(deploy_path))
    print(f"Saved selected model for deployment to: {deploy_path}")

    # Save comparison summary
    comparison_summary = {
        "dataset": "PlantDoc",
        "clean_train_images": len(train_paths),
        "val_images": len(val_paths),
        "mobilenetv2_val": mobilenet_val,
        "efficientnetb0_val": effnet_val,
        "selected_model": selected_name,
        "deploy_model_path": str(deploy_path)
    }
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_summary, f, indent=4)

    print("\nTraining and model selection completed successfully.")
    print(f"Report saved to: {REPORTS_DIR / 'model_comparison.json'}")

if __name__ == "__main__":
    main()
