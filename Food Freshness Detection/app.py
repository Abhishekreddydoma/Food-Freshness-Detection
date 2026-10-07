"""
app.py
------
Flask web application for Food Freshness Detection.
Loads the trained EfficientNet-B0 model once at startup and serves:
  - GET  /            -> Web UI
  - POST /predict     -> Real-time freshness classification & Grad-CAM generation
  - GET  /report/csv  -> Download CSV report for the most recent analysis
  - GET  /report/pdf  -> Download PDF report for the most recent analysis
"""

from datetime import datetime
import io
import json
import os
import sys
from pathlib import Path
from PIL import Image

# Suppress TF info/warning logs before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, render_template, request, send_file

from gradcam.gradcam import generate_gradcam
from reports.report_generator import generate_csv_report, generate_pdf_report

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "model" / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"

GRADCAM_OUTPUT_DIR = PROJECT_ROOT / "static" / "gradcam"
GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated"
REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# ---------------------------------------------------------------------------
# Global Model, Class Names & Latest Analysis Cache
# ---------------------------------------------------------------------------
model = None
class_names = []
latest_analysis = None


def load_resources():
    """Load class mapping and trained model once during startup."""
    global model, class_names

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_names = data.get("class_names", [])
    if not class_names:
        raise ValueError(f"Empty or invalid class list in {CLASS_NAMES_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model file not found: {MODEL_PATH}")
    print(f"[*] Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[OK] Model loaded successfully ({len(class_names)} classes).")


# Load model and class names upon module import / server startup
load_resources()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_allowed_extension(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def parse_produce_and_freshness(class_name: str):
    """
    Derive produce type and freshness condition from the class name.
    Examples:
      FreshApple -> ('Apple', 'Fresh')
      RottenApple -> ('Apple', 'Rotten')
      FreshTomato -> ('Tomato', 'Fresh')
    """
    if class_name.startswith("Fresh"):
        return class_name[len("Fresh"):], "Fresh"
    elif class_name.startswith("Rotten"):
        return class_name[len("Rotten"):], "Rotten"
    return class_name, "Unknown"


def derive_quality_inspection(produce: str, freshness: str):
    """
    Derive quality status and visual inspection summary based strictly
    on the visual classification of produce and freshness.
    """
    if freshness == "Fresh":
        quality_status = "Visually appears fresh"
        visual_summary = f"The model detected visual characteristics associated with fresh {produce}."
    elif freshness == "Rotten":
        quality_status = "Signs of spoilage detected"
        visual_summary = f"The model detected visual characteristics associated with spoiled {produce}."
    else:
        quality_status = "Unknown quality status"
        visual_summary = f"Visual characteristics for {produce} could not be determined."
    return quality_status, visual_summary


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes matching the training pipeline:
    - Verify image integrity
    - Convert to RGB
    - Resize to (IMG_SIZE, IMG_SIZE)
    - Return float32 array in [0, 255] with batch dimension (1, 224, 224, 3)
    """
    # Verify image integrity
    with Image.open(io.BytesIO(image_bytes)) as img_check:
        img_check.verify()

    # Open image for processing
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)  # [0, 255] range
        img_tensor = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
        return img_tensor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the homepage. Fast response, no heavy operations."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle food image prediction and Grad-CAM generation:
    - Accepts multipart/form-data with file field 'food_image' or 'file'
    - Validates file format and content integrity
    - Performs model inference
    - Generates real Grad-CAM visual explanation
    - Stores latest analysis result in memory for report generation
    - Returns JSON with prediction, produce, freshness, confidence, gradcam_image, quality_status, and visual_summary
    """
    global latest_analysis

    # 1. Check if model is initialized
    if model is None or not class_names:
        return jsonify({
            "success": False,
            "error": "Model not initialized or unavailable."
        }), 500

    # 2. Check for uploaded file
    file = None
    if "food_image" in request.files:
        file = request.files["food_image"]
    elif "file" in request.files:
        file = request.files["file"]

    if file is None or file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose an image to upload."
        }), 400

    # 3. Validate file extension
    if not is_allowed_extension(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload a JPG, JPEG, or PNG image."
        }), 400

    # 4. Read file bytes and validate content
    try:
        file_bytes = file.read()
        if not file_bytes or len(file_bytes) == 0:
            return jsonify({
                "success": False,
                "error": "Empty file uploaded. Please upload a valid image."
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to read uploaded file: {str(e)}"
        }), 400

    # 5. Preprocess image
    try:
        preprocessed_tensor = preprocess_image_bytes(file_bytes)
    except Exception:
        return jsonify({
            "success": False,
            "error": "Unreadable or corrupted image file. Please provide a valid image."
        }), 400

    # 6. Run model prediction
    try:
        predictions = model.predict(preprocessed_tensor, verbose=0)
        probabilities = predictions[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        produce, freshness = parse_produce_and_freshness(predicted_class)
        quality_status, visual_summary = derive_quality_inspection(produce, freshness)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

    # 7. Generate Grad-CAM explanation for the predicted class
    gradcam_image_url = None
    gradcam_error = None
    try:
        saved_gradcam_path = generate_gradcam(
            model=model,
            image_bytes=file_bytes,
            class_index=predicted_idx,
            output_dir=GRADCAM_OUTPUT_DIR,
        )
        gradcam_filename = Path(saved_gradcam_path).name
        gradcam_image_url = f"/static/gradcam/{gradcam_filename}"
    except Exception as e:
        gradcam_error = f"Grad-CAM generation failed: {str(e)}"

    # 8. Cache latest successful analysis for report generation
    latest_analysis = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": Path(file.filename).name,
        "predicted_class": predicted_class,
        "produce": produce,
        "freshness": freshness,
        "confidence": confidence,
        "quality_status": quality_status,
        "visual_summary": visual_summary,
        "gradcam_image": gradcam_image_url,
    }

    response_payload = {
        "success": True,
        "predicted_class": predicted_class,
        "produce": produce,
        "freshness": freshness,
        "confidence": confidence,
        "gradcam_image": gradcam_image_url,
        "quality_status": quality_status,
        "visual_summary": visual_summary,
    }
    if gradcam_error:
        response_payload["gradcam_error"] = gradcam_error

    return jsonify(response_payload), 200


@app.route("/report/csv", methods=["GET"])
def download_csv_report():
    """
    Generate and serve a CSV report for the most recent successful analysis.
    Does NOT rerun model prediction.
    """
    if latest_analysis is None:
        return jsonify({"error": "No analysis available. Please analyze an image first."}), 400

    try:
        csv_file_path = generate_csv_report(latest_analysis, REPORTS_OUTPUT_DIR)
        return send_file(
            csv_file_path,
            as_attachment=True,
            download_name=Path(csv_file_path).name,
            mimetype="text/csv",
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate CSV report: {str(e)}"}), 500


@app.route("/report/pdf", methods=["GET"])
def download_pdf_report():
    """
    Generate and serve a PDF report for the most recent successful analysis.
    Does NOT rerun model prediction.
    """
    if latest_analysis is None:
        return jsonify({"error": "No analysis available. Please analyze an image first."}), 400

    try:
        pdf_file_path = generate_pdf_report(
            latest_analysis,
            REPORTS_OUTPUT_DIR,
            project_root=PROJECT_ROOT,
        )
        return send_file(
            pdf_file_path,
            as_attachment=True,
            download_name=Path(pdf_file_path).name,
            mimetype="application/pdf",
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
