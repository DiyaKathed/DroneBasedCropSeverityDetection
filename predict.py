"""
Prediction CLI script for the PlantDoc Disease Classification Model.
Loads an image, prepares raw RGB float32 input [0, 255] (matching the model's
internal Rescaling layer), and outputs the top predicted disease classes.
"""

import sys
import json
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image

MODEL_PATH = "models/plantdoc_disease_model.keras"
LABELS_PATH = "class_labels.json"

def load_resources(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels not found: {labels_path}")

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        class_names = [labels[str(i)] for i in range(len(labels))]
    else:
        class_names = labels
    return model, class_names

def predict_image(image_path, model, class_names, top_k=3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize((224, 224))
        image_array = np.array(img_rgb, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

    # Note: Model contains internal Rescaling(scale=1/127.5, offset=-1.0)
    # Feeding raw float32 [0, 255] is required.
    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "disease": class_names[idx],
            "confidence": round(float(predictions[idx]) * 100, 2)
        })
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image> [--json]")
        sys.exit(1)

    image_path = sys.argv[1]
    is_json = "--json" in sys.argv

    try:
        model, class_names = load_resources()
        results = predict_image(image_path, model, class_names)
        
        # Run computer vision severity estimation
        from severity_estimator import estimate_severity
        severity_result = estimate_severity(image_path)

        # Load treatment recommendations from database
        treatment_info = None
        db_path = "disease_database.json"
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as dbf:
                    db_data = json.load(dbf)
                class_map = {d["class_name"].strip().lower(): d for d in db_data.get("diseases", [])}
                treatment_info = class_map.get(results[0]["disease"].strip().lower())
            except Exception:
                treatment_info = None

        if is_json:
            print(json.dumps({
                "status": "success",
                "image": image_path,
                "top_predictions": results,
                "predicted_disease": results[0]["disease"],
                "confidence": results[0]["confidence"],
                "severity_assessment": severity_result,
                "disease_info": {
                    "crop": treatment_info.get("crop") if treatment_info else None,
                    "scientific_name": treatment_info.get("scientific_name") if treatment_info else None,
                    "pathogen_type": treatment_info.get("pathogen_type") if treatment_info else None,
                    "description": treatment_info.get("description") if treatment_info else None
                } if treatment_info else None,
                "treatment_and_management": treatment_info.get("treatment_and_management") if treatment_info else None
            }, indent=2))
        else:
            print("\nPlant Disease Prediction, Severity & Treatment Report")
            print("==================================================")
            print(f"Image: {image_path}")
            print(f"Top Disease: {results[0]['disease']} ({results[0]['confidence']}%)")
            if treatment_info:
                print(f"Crop:        {treatment_info.get('crop')} | Type: {treatment_info.get('pathogen_type')} | Scientific: {treatment_info.get('scientific_name')}")
            print("\nTop Differential Diagnoses:")
            for rank, item in enumerate(results, 1):
                print(f"  {rank}. {item['disease']:<38} {item['confidence']:>6.2f}%")
            print("--------------------------------------------------")
            print(f"Severity Status: {severity_result['status']}")
            if severity_result.get("level"):
                print(f"Visual Damage Level: {severity_result['level']}")
                print(f"Affected Leaf Area:  {severity_result['affected_area_percentage']}%")
                print(f"Leaf Coverage Ratio: {severity_result.get('leaf_coverage_percentage', 'N/A')}%")
            print(f"Severity Note:   {severity_result['explanation']}")
            print(f"Disclaimer:      {severity_result.get('disclaimer', '')}")
            
            if treatment_info and treatment_info.get("treatment_and_management"):
                tm = treatment_info["treatment_and_management"]
                print("--------------------------------------------------")
                print("TREATMENT & MANAGEMENT RECOMMENDATIONS:")
                if tm.get("immediate_actions"):
                    print("\n* Immediate Actions:")
                    for act in tm["immediate_actions"]:
                        print(f"  - {act}")
                if tm.get("cultural_practices"):
                    print("\n* Cultural & Preventive Practices:")
                    for cp in tm["cultural_practices"]:
                        print(f"  - {cp}")
                if tm.get("treatment_options"):
                    print("\n* Recommended Treatment Options:")
                    for to in tm["treatment_options"]:
                        print(f"  - {to}")
                if tm.get("expert_advice"):
                    print(f"\n* Expert Consultation: {tm['expert_advice']}")
                if tm.get("disclaimer"):
                    print(f"\n* Safety Notice: {tm['disclaimer']}")

            print("==================================================")
            print("Note: Prediction confidence reflects model certainty, not infection severity.\n")

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()