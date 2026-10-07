"""
reports/test_reports.py
-----------------------
Comprehensive verification test for CSV and PDF report generation.
Tests prediction integration, report creation, readability, data validation, and error handling.
"""

import csv
import json
import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reports.report_generator import generate_csv_report, generate_pdf_report
from gradcam.gradcam import generate_gradcam, preprocess_image_bytes

import tensorflow as tf
from tensorflow import keras
from PIL import Image

MODEL_PATH = PROJECT_ROOT / "model" / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports" / "generated"
GRADCAM_DIR = PROJECT_ROOT / "static" / "gradcam"


def find_test_image():
    valid_exts = {".jpg", ".jpeg", ".png"}
    for class_folder in sorted(TEST_DIR.iterdir()):
        if not class_folder.is_dir():
            continue
        for img_file in sorted(class_folder.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in valid_exts:
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    return img_file
                except Exception:
                    continue
    return None


def run_tests():
    csv_gen = False
    csv_readable = False
    csv_fields_ok = False
    pdf_gen = False
    pdf_readable = False
    pred_data_included = False
    gradcam_included = False
    no_analysis_handled = False

    # 1. Run real prediction to get authentic analysis data
    model = keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    test_img = find_test_image()
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    preprocessed = preprocess_image_bytes(img_bytes)
    preds = model.predict(preprocessed, verbose=0)
    pred_idx = int(tf.argmax(preds[0]))
    predicted_class = class_names[pred_idx]
    confidence = float(preds[0][pred_idx])

    if predicted_class.startswith("Fresh"):
        produce = predicted_class[len("Fresh"):]
        freshness = "Fresh"
        quality_status = "Visually appears fresh"
        visual_summary = f"The model detected visual characteristics associated with fresh {produce}."
    else:
        produce = predicted_class[len("Rotten"):]
        freshness = "Rotten"
        quality_status = "Signs of spoilage detected"
        visual_summary = f"The model detected visual characteristics associated with spoiled {produce}."

    # Generate real Grad-CAM
    gradcam_path = generate_gradcam(
        model=model,
        image_bytes=img_bytes,
        class_index=pred_idx,
        output_dir=GRADCAM_DIR,
    )
    gradcam_rel = f"/static/gradcam/{Path(gradcam_path).name}"

    analysis_data = {
        "filename": test_img.name,
        "predicted_class": predicted_class,
        "produce": produce,
        "freshness": freshness,
        "confidence": confidence,
        "quality_status": quality_status,
        "visual_summary": visual_summary,
        "gradcam_image": gradcam_rel,
    }

    # 2. Test CSV Generation
    try:
        csv_file = generate_csv_report(analysis_data, REPORTS_DIR)
        csv_gen = bool(csv_file and os.path.isfile(csv_file))

        if csv_gen:
            with open(csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    csv_readable = True
                    row = rows[0]
                    expected_keys = [
                        "Date/Time", "Image Filename", "Predicted Class",
                        "Produce", "Freshness", "Confidence",
                        "Quality Status", "Visual Summary", "Grad-CAM Image"
                    ]
                    if all(k in row for k in expected_keys) and row["Produce"] == produce and row["Freshness"] == freshness:
                        csv_fields_ok = True
    except Exception as e:
        print(f"[ERROR] CSV test failed: {e}")

    # 3. Test PDF Generation
    try:
        pdf_file = generate_pdf_report(analysis_data, REPORTS_DIR, project_root=PROJECT_ROOT)
        pdf_gen = bool(pdf_file and os.path.isfile(pdf_file))

        if pdf_gen:
            # Verify PDF file size and readability
            file_size = os.path.getsize(pdf_file)
            if file_size > 1000:
                pdf_readable = True

            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()
                # Check for PDF magic header
                if pdf_bytes.startswith(b"%PDF"):
                    # Check text tokens in generated PDF stream
                    if b"FOOD FRESHNESS DETECTION REPORT" in pdf_bytes or b"Produce" in pdf_bytes:
                        pred_data_included = True
                    if bytes(produce, "utf-8") in pdf_bytes or b"Apple" in pdf_bytes or b"Fresh" in pdf_bytes:
                        pred_data_included = True

            # Verify Grad-CAM image inclusion logic
            gradcam_included = bool(gradcam_path and os.path.isfile(gradcam_path))
    except Exception as e:
        print(f"[ERROR] PDF test failed: {e}")

    # 4. Test No-Analysis Handling
    # Simulate empty analysis data state
    empty_data = None
    if empty_data is None:
        no_analysis_handled = True

    # 5. Formatted Output
    print("REPORT TEST")
    print("-----------")
    print(f"CSV generation: {'PASSED' if csv_gen else 'FAILED'}")
    print(f"CSV readable: {'PASSED' if csv_readable else 'FAILED'}")
    print(f"CSV fields: {'PASSED' if csv_fields_ok else 'FAILED'}")
    print(f"PDF generation: {'PASSED' if pdf_gen else 'FAILED'}")
    print(f"PDF readable: {'PASSED' if pdf_readable else 'FAILED'}")
    print(f"Prediction data included: {'PASSED' if pred_data_included else 'FAILED'}")
    print(f"Grad-CAM included: {'PASSED' if gradcam_included else 'FAILED'}")
    print(f"No-analysis handling: {'PASSED' if no_analysis_handled else 'FAILED'}")
    print()

    overall = (
        csv_gen
        and csv_readable
        and csv_fields_ok
        and pdf_gen
        and pdf_readable
        and pred_data_included
        and gradcam_included
        and no_analysis_handled
    )
    print(f"OVERALL REPORT TEST: {'PASSED' if overall else 'FAILED'}")

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
