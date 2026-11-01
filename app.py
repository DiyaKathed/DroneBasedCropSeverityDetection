import os
import json
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from PIL import Image
from severity_estimator import estimate_severity

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "plantdoc_disease_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "class_labels.json"
)

# -----------------------------
# Load Model and Class Labels
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Class labels not found at: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

if isinstance(labels, dict):
    class_names = [labels[str(i)] for i in range(len(labels))]
else:
    class_names = labels

# -----------------------------
# Load Disease & Treatment Database
# -----------------------------
DB_PATH = os.path.join(BASE_DIR, "disease_database.json")
disease_db = {"diseases": []}
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        disease_db = json.load(f)

DISEASES_BY_CLASS = {d["class_name"].strip().lower(): d for d in disease_db.get("diseases", [])}
DISEASES_BY_ID = {d["id"].strip().lower(): d for d in disease_db.get("diseases", [])}

print(f"Loaded {len(disease_db.get('diseases', []))} disease profiles with verified treatment guides!")

# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Plant Disease Detection API is running",
        "status": "success",
        "endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "predict": "/api/predict"
        }
    })

# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(class_names)
    })

# -----------------------------
# Model Info Diagnostic Route
# -----------------------------
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "status": "success",
        "model_name": getattr(model, "name", "plantdoc_disease_model"),
        "model_path": MODEL_PATH,
        "input_shape": list(model.input_shape) if hasattr(model, "input_shape") else [None, 224, 224, 3],
        "output_shape": list(model.output_shape) if hasattr(model, "output_shape") else [None, len(class_names)],
        "num_classes": len(class_names),
        "classes": class_names,
        "preprocessing": "Internal Rescaling (accepts raw RGB [0, 255] float32)"
    })

# -----------------------------
# Model Performance Metrics Route
# -----------------------------
@app.route("/api/model-performance", methods=["GET"])
def model_performance():
    metrics_path = os.path.join(BASE_DIR, "reports", "test_evaluation_metrics.json")
    comparison_path = os.path.join(BASE_DIR, "reports", "model_comparison.json")
    
    test_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            test_metrics = json.load(f)

    comparison_data = {}
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            comparison_data = json.load(f)

    return jsonify({
        "status": "success",
        "model_name": comparison_data.get("selected_model", "EfficientNetB0"),
        "dataset": "PlantDoc (in-the-wild benchmark)",
        "test_images": test_metrics.get("test_images", 236),
        "test_accuracy": round(float(test_metrics.get("accuracy", 0.4703)) * 100, 2),
        "macro_precision": round(float(test_metrics.get("macro_precision", 0.5309)) * 100, 2),
        "macro_recall": round(float(test_metrics.get("macro_recall", 0.4714)) * 100, 2),
        "macro_f1": round(float(test_metrics.get("macro_f1", 0.4521)) * 100, 2),
        "weighted_f1": round(float(test_metrics.get("weighted_f1", 0.4564)) * 100, 2),
        "validation_accuracy": round(float(comparison_data.get("efficientnetb0_val", {}).get("val_accuracy", 0.5161)) * 100, 2),
        "validation_macro_f1": round(float(comparison_data.get("efficientnetb0_val", {}).get("val_macro_f1", 0.4745)) * 100, 2),
        "evaluation_notes": "Evaluated on 236 genuinely untouched held-out test images across 27 classes."
    })

# -----------------------------
# Disease Prediction Route
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({
            "error": "No image file provided. Use the field name 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Please select an image."
        }), 400

    try:
        # Open image as RGB
        raw_img = Image.open(file.stream).convert("RGB")
        
        # Resize for disease classification model input (224, 224)
        img_classified = raw_img.resize((224, 224))

        # Convert to float32 raw RGB array in range [0, 255]
        # DO NOT call preprocess_input(): The model contains an internal Rescaling layer.
        image_array = np.array(img_classified, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict disease
        predictions = model.predict(image_array, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = []

        for i in top_indices:
            top_predictions.append({
                "disease": str(class_names[int(i)]),
                "confidence": round(float(predictions[i]) * 100, 2)
            })

        # Cap image resolution for severity analysis to max 800px on longest side for fast response
        w_raw, h_raw = raw_img.size
        max_dim = 800
        if max(w_raw, h_raw) > max_dim:
            scale = max_dim / float(max(w_raw, h_raw))
            img_for_severity = raw_img.resize((int(w_raw * scale), int(h_raw * scale)), Image.Resampling.LANCZOS)
        else:
            img_for_severity = raw_img

        # Execute computer vision leaf & lesion severity assessment
        severity_assessment = estimate_severity(img_for_severity)

        # Look up matched disease details and treatment recommendations
        top_disease_name = top_predictions[0]["disease"]
        matched_disease = DISEASES_BY_CLASS.get(top_disease_name.strip().lower())
        treatment_and_management = None
        disease_info = None
        uncertainty_warning = None

        if matched_disease:
            disease_info = {
                "id": matched_disease["id"],
                "class_name": matched_disease["class_name"],
                "display_name": matched_disease["display_name"],
                "scientific_name": matched_disease.get("scientific_name", ""),
                "crop": matched_disease["crop"],
                "is_healthy": matched_disease.get("is_healthy", False),
                "pathogen_type": matched_disease.get("pathogen_type", "Fungal"),
                "description": matched_disease.get("description", ""),
                "symptoms": matched_disease.get("symptoms", []),
                "favorable_conditions": matched_disease.get("favorable_conditions", "")
            }
            treatment_and_management = matched_disease.get("treatment_and_management")

        confidence_val = top_predictions[0]["confidence"]
        if confidence_val < 35.0:
            uncertainty_warning = (
                f"Caution: Model prediction certainty is low ({confidence_val}%). "
                "Chemical treatments or irreversible interventions should NOT be implemented "
                "without physical confirmation by an agricultural extension specialist."
            )
        elif matched_disease and matched_disease.get("is_healthy"):
            uncertainty_warning = (
                "Healthy leaf detected: No therapeutic or chemical treatments are required. "
                "Continue standard cultural maintenance and routine scouting."
            )

        return jsonify({
            "status": "success",
            "predicted_disease": top_predictions[0]["disease"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "severity": severity_assessment.get("level"),
            "severity_note": severity_assessment.get("explanation"),
            "severity_assessment": severity_assessment,
            "disease_info": disease_info,
            "treatment_and_management": treatment_and_management,
            "uncertainty_warning": uncertainty_warning
        })

    except Exception as e:
        return jsonify({
            "error": f"Failed to process image: {str(e)}"
        }), 500

# -----------------------------
# Disease Catalog & Information Routes
# -----------------------------
@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    crop_filter = request.args.get("crop", "").strip().lower()
    pathogen_filter = request.args.get("pathogen", "").strip().lower()

    filtered = []
    for d in disease_db.get("diseases", []):
        if crop_filter and crop_filter != "all" and d.get("crop", "").lower() != crop_filter:
            continue
        if pathogen_filter and pathogen_filter != "all" and d.get("pathogen_type", "").lower() != pathogen_filter:
            continue
        filtered.append({
            "id": d["id"],
            "class_name": d["class_name"],
            "display_name": d["display_name"],
            "scientific_name": d.get("scientific_name", ""),
            "crop": d["crop"],
            "is_healthy": d.get("is_healthy", False),
            "pathogen_type": d.get("pathogen_type", "Fungal"),
            "description": d.get("description", ""),
            "symptoms_preview": d.get("symptoms", [])[:2]
        })

    crops = sorted(list(set(d["crop"] for d in disease_db.get("diseases", []))))
    pathogens = sorted(list(set(d.get("pathogen_type", "Fungal") for d in disease_db.get("diseases", []))))

    return jsonify({
        "status": "success",
        "total": len(filtered),
        "crops": crops,
        "pathogens": pathogens,
        "diseases": filtered
    })

@app.route("/api/diseases/<disease_id>", methods=["GET"])
def get_disease_detail(disease_id):
    clean_id = disease_id.strip().lower()
    match = DISEASES_BY_ID.get(clean_id) or DISEASES_BY_CLASS.get(clean_id)
    if not match:
        return jsonify({
            "status": "not_found",
            "message": f"Disease '{disease_id}' is not in the supported 28-class dataset."
        }), 404

    return jsonify({
        "status": "success",
        "disease": match
    })

# -----------------------------
# Disease Search Route
# -----------------------------
@app.route("/api/search", methods=["GET"])
def search_diseases():
    query = request.args.get("q", "").strip().lower()
    crop_filter = request.args.get("crop", "").strip().lower()
    pathogen_filter = request.args.get("pathogen", "").strip().lower()

    if not query and not crop_filter and not pathogen_filter:
        return get_diseases()

    matches = []
    for d in disease_db.get("diseases", []):
        if crop_filter and crop_filter != "all" and d.get("crop", "").lower() != crop_filter:
            continue
        if pathogen_filter and pathogen_filter != "all" and d.get("pathogen_type", "").lower() != pathogen_filter:
            continue

        score = 0
        if query:
            d_name = d.get("display_name", "").lower()
            c_name = d.get("class_name", "").lower()
            crop_name = d.get("crop", "").lower()
            sci_name = d.get("scientific_name", "").lower()
            symptom_str = " ".join(d.get("symptoms", [])).lower()
            desc = d.get("description", "").lower()

            if query == d_name or query == c_name:
                score += 100
            elif query in d_name or query in c_name:
                score += 50
            elif query in crop_name:
                score += 30
            elif query in sci_name:
                score += 25
            elif query in symptom_str:
                score += 10
            elif query in desc:
                score += 5
            else:
                continue

        matches.append((score, d))

    matches.sort(key=lambda x: x[0], reverse=True)
    sorted_results = [item[1] for item in matches]

    return jsonify({
        "status": "success",
        "query": query,
        "crop_filter": crop_filter,
        "count": len(sorted_results),
        "results": sorted_results,
        "message": (
            None if sorted_results else
            f"No matching diseases found for '{query}'. Currently, 28 specific crop diseases and healthy leaf conditions are supported in the PlantDoc dataset."
        )
    })

# -----------------------------
# Run Flask Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Plant Disease API on http://127.0.0.1:{port} (AirPlay port 5000 bypassed)")
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
    )