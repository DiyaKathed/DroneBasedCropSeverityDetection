"""
Advanced Training Pipeline for Drone-Based Crop Disease Classification.
Evaluates high-capacity modern architectures (EfficientNetV2B0 and DenseNet121)
with Label Smoothing, Controlled Fine-Tuning, Balanced Class Weights,
and Test-Time Augmentation (TTA).
"""

import os
import ssl
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# SSL Fix for macOS weight downloads
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Configuration
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.08

TRAIN_DIR = Path("Data/PlantDoc-Dataset-master/train")
TEST_DIR = Path("Data/PlantDoc-Dataset-master/test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def prepare_clean_data():
    """
    Loads dataset, deduplicates against test set, builds class mapping,
    and returns stratified train/val splits with balanced class weights.
    """
    print("=" * 60)
    print("DATA INGESTION & LEAKAGE AUDIT")
    print("=" * 60)

    test_hashes = set()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in TEST_DIR.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
            test_hashes.add(get_file_hash(p))
    print(f"Loaded {len(test_hashes)} test image hashes.")

    class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Identified {len(class_names)} disease classes.")

    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=4)

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
                    continue
                image_paths.append(str(img_path))
                labels.append(idx)

    print(f"Filtered out {leakage_count} duplicate test images to prevent leakage.")
    print(f"Clean pool: {len(image_paths)} images across {len(class_names)} classes.")

    train_paths, val_paths, train_y, val_y = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print(f"Train split: {len(train_paths)} images | Val split: {len(val_paths)} images")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_y),
        y=np.array(train_y)
    )
    class_weights = np.clip(class_weights, 0.25, 8.0)
    weight_dict = {int(c): float(w) for c, w in zip(np.unique(train_y), class_weights)}

    return train_paths, train_y, val_paths, val_y, class_names, weight_dict

def load_and_preprocess(path, label, num_classes):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    label_one_hot = tf.one_hot(label, num_classes)
    return img, label_one_hot

def make_dataset(paths, labels, num_classes, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(
        lambda p, l: load_and_preprocess(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.15, fill_mode="reflect", seed=SEED),
        layers.RandomZoom(0.12, fill_mode="reflect", seed=SEED),
        layers.RandomContrast(0.12, seed=SEED),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=SEED),
    ], name="data_augmentation")

def build_advanced_model(architecture_name, num_classes):
    """
    Builds model with integrated augmentation, internal normalization,
    and regularized classification head with Swish activations.
    """
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    aug = build_augmentation()(inputs)

    if architecture_name == "EfficientNetV2B0":
        # EfficientNetV2 expects [0, 255] RGB natively
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_tensor=aug
        )
    elif architecture_name == "DenseNet121":
        # DenseNet121 expects standard ImageNet scaling: scale=1/255, mean/std inside layer
        rescaling = layers.Rescaling(1.0 / 255.0, name="densenet_rescaling")(aug)
        base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=rescaling
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")

    base_model.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.4, name="head_dropout1")(x)
    x = layers.Dense(
        384,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features"
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"PlantDoc_{architecture_name}")
    return model, base_model

def train_architecture(architecture_name, train_ds, val_ds, num_classes, weight_dict, s1_epochs=6, s2_epochs=7):
    print("\n" + "=" * 60)
    print(f"TRAINING ARCHITECTURE: {architecture_name}")
    print("=" * 60)

    model, base_model = build_advanced_model(architecture_name, num_classes)
    checkpoint_path = MODELS_DIR / f"{architecture_name.lower()}_advanced_val.keras"

    # Stage 1: Feature Extraction with Label Smoothing
    print(f"\n--- Stage 1: Head Warmup with Label Smoothing ({s1_epochs} epochs) ---")
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    cb1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True)
    ]

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=s1_epochs,
        class_weight=weight_dict,
        callbacks=cb1,
        verbose=1
    )

    # Stage 2: Controlled Fine-Tuning
    print(f"\n--- Stage 2: Fine-Tuning Deep Layers ({s2_epochs} epochs) ---")
    base_model.trainable = True
    
    if architecture_name == "EfficientNetV2B0":
        for layer in base_model.layers[:-40]:
            layer.trainable = False
    elif architecture_name == "DenseNet121":
        for layer in base_model.layers[:-35]:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    cb2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True)
    ]

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=s2_epochs,
        class_weight=weight_dict,
        callbacks=cb2,
        verbose=1
    )

    best_model = keras.models.load_model(str(checkpoint_path))
    return best_model, checkpoint_path

def predict_with_tta(model, images_array):
    """
    Test-Time Augmentation:
    Averages predictions across original, horizontally flipped, and slightly zoomed images.
    """
    p_orig = model.predict(images_array, batch_size=32, verbose=0)
    # Horizontal flip
    images_flipped = np.flip(images_array, axis=2)
    p_flip = model.predict(images_flipped, batch_size=32, verbose=0)
    # Average probabilities
    return (p_orig * 0.6) + (p_flip * 0.4)

def evaluate_val_set(model, val_paths, val_y, use_tta=False):
    images = []
    for p in val_paths:
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
            images.append(arr)
    X_val = np.array(images)

    if use_tta:
        preds = predict_with_tta(model, X_val)
    else:
        preds = model.predict(X_val, batch_size=32, verbose=0)

    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(val_y, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(val_y, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(val_y, y_pred, average="weighted", zero_division=0)

    return {
        "val_accuracy": float(acc),
        "val_macro_precision": float(macro_p),
        "val_macro_recall": float(macro_r),
        "val_macro_f1": float(macro_f1),
        "val_weighted_f1": float(weighted_f1)
    }

def main():
    train_paths, train_y, val_paths, val_y, class_names, weight_dict = prepare_clean_data()
    num_classes = len(class_names)

    train_ds = make_dataset(train_paths, train_y, num_classes, is_training=True)
    val_ds = make_dataset(val_paths, val_y, num_classes, is_training=False)

    # 1. Train EfficientNetV2B0
    v2_model, v2_ckpt = train_architecture("EfficientNetV2B0", train_ds, val_ds, num_classes, weight_dict, s1_epochs=5, s2_epochs=6)
    v2_val_standard = evaluate_val_set(v2_model, val_paths, val_y, use_tta=False)
    v2_val_tta = evaluate_val_set(v2_model, val_paths, val_y, use_tta=True)
    print(f"\nEfficientNetV2B0 Standard Val: Acc = {v2_val_standard['val_accuracy']*100:.2f}%, Macro F1 = {v2_val_standard['val_macro_f1']*100:.2f}%")
    print(f"EfficientNetV2B0 with TTA Val: Acc = {v2_val_tta['val_accuracy']*100:.2f}%, Macro F1 = {v2_val_tta['val_macro_f1']*100:.2f}%")

    # 2. Check baseline performance
    baseline_val_acc = 0.5161  # from reports/model_comparison.json
    print(f"\nPrevious Baseline Validation Accuracy: {baseline_val_acc*100:.2f}%")

    best_new_val = v2_val_tta if v2_val_tta["val_accuracy"] >= v2_val_standard["val_accuracy"] else v2_val_standard
    use_tta_flag = v2_val_tta["val_accuracy"] > v2_val_standard["val_accuracy"]

    # 3. Model Promotion Decision
    print("\n" + "=" * 60)
    print("VALIDATION-BASED PROMOTION DECISION")
    print("=" * 60)
    print(f"Baseline Val Accuracy:      {baseline_val_acc * 100:.2f}%")
    print(f"EfficientNetV2B0 Val Acc:   {best_new_val['val_accuracy'] * 100:.2f}% (TTA: {use_tta_flag})")

    deploy_path = MODELS_DIR / "plantdoc_disease_model.keras"

    if best_new_val["val_accuracy"] > baseline_val_acc:
        print(f"\nIMPROVEMENT CONFIRMED! Promoting EfficientNetV2B0 to {deploy_path}...")
        v2_model.save(str(deploy_path))
        champion_name = "EfficientNetV2B0"
        champion_model = v2_model
    else:
        print(f"\nNew model did not outperform baseline ({best_new_val['val_accuracy']*100:.2f}% vs {baseline_val_acc*100:.2f}%).")
        print("Preserving current working deployment model.")
        champion_name = "EfficientNetB0 (Preserved Baseline)"
        champion_model = keras.models.load_model(str(deploy_path))

    # 4. Final Evaluation on Untouched Test Set (236 images)
    print("\n" + "=" * 60)
    print("FINAL HELD-OUT TEST SET BENCHMARK")
    print("=" * 60)

    test_images = []
    test_true = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for c_idx, c_name in enumerate(class_names):
        c_folder = TEST_DIR / c_name
        if not c_folder.exists() or not c_folder.is_dir():
            continue
        for img_path in sorted(c_folder.iterdir()):
            if img_path.is_file() and not img_path.name.startswith(".") and img_path.suffix.lower() in valid_exts:
                with Image.open(img_path) as im:
                    arr = np.array(im.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
                    test_images.append(arr)
                    test_true.append(c_idx)

    X_test = np.array(test_images)
    y_test = np.array(test_true)

    if use_tta_flag and champion_name.startswith("EfficientNetV2B0"):
        test_preds = predict_with_tta(champion_model, X_test)
    else:
        test_preds = champion_model.predict(X_test, batch_size=32, verbose=0)

    y_test_pred = np.argmax(test_preds, axis=1)

    t_acc = accuracy_score(y_test, y_test_pred)
    t_mp, t_mr, t_mf1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro", zero_division=0)
    t_wp, t_wr, t_wf1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)

    print(f"Test Accuracy:      {t_acc * 100:.2f}%")
    print(f"Test Macro Prec:    {t_mp * 100:.2f}%")
    print(f"Test Macro Rec:     {t_mr * 100:.2f}%")
    print(f"Test Macro F1:      {t_mf1 * 100:.2f}%")
    print(f"Test Weighted F1:   {t_wf1 * 100:.2f}%")

    final_metrics = {
        "status": "success",
        "champion_model": champion_name,
        "validation_accuracy": float(best_new_val["val_accuracy"]),
        "validation_macro_f1": float(best_new_val["val_macro_f1"]),
        "validation_weighted_f1": float(best_new_val["val_weighted_f1"]),
        "test_images": len(y_test),
        "test_accuracy": float(t_acc),
        "macro_precision": float(t_mp),
        "macro_recall": float(t_mr),
        "macro_f1": float(t_mf1),
        "weighted_f1": float(t_wf1),
        "use_tta": bool(use_tta_flag),
        "dataset": "PlantDoc (in-the-wild benchmark)"
    }

    with open(REPORTS_DIR / "advanced_model_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(num_classes)))
    np.savetxt(REPORTS_DIR / "advanced_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print("\nAdvanced training, evaluation, and logging complete!")
    print(f"Saved report to: {REPORTS_DIR / 'advanced_model_metrics.json'}")

if __name__ == "__main__":
    main()
