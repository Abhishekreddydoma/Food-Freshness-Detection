"""
gradcam/test_gradcam.py
-----------------------
Test script for the real Grad-CAM implementation.
Loads the trained EfficientNet-B0 model, picks a real test image from dataset/test/,
computes real prediction & Grad-CAM heatmap, saves the overlay image, and verifies it.
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image

# Suppress TF info/warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow import keras

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradcam.gradcam import generate_gradcam, preprocess_image_bytes

MODEL_PATH = PROJECT_ROOT / "model" / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "static" / "gradcam"


def find_valid_test_image(test_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    for class_folder in sorted(test_dir.iterdir()):
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


def run_test():
    # 1. Load model and class names
    if not MODEL_PATH.is_file():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)

    model = keras.models.load_model(str(MODEL_PATH))
    model_loaded = True

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    # 2. Select one real image
    test_img_path = find_valid_test_image(TEST_DIR)
    if not test_img_path:
        print("[ERROR] No valid test image found.")
        sys.exit(1)

    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    # 3. Predict class
    preprocessed = preprocess_image_bytes(img_bytes)
    preds = model.predict(preprocessed, verbose=0)
    pred_idx = int(tf.argmax(preds[0]))
    predicted_class = class_names[pred_idx]

    # 4. Generate Grad-CAM for predicted class
    output_path = generate_gradcam(
        model=model,
        image_bytes=img_bytes,
        class_index=pred_idx,
        output_dir=OUTPUT_DIR,
    )
    gradcam_generated = bool(output_path and os.path.exists(output_path))

    # 5. Verify image is readable
    output_readable = False
    if gradcam_generated:
        try:
            with Image.open(output_path) as img:
                img.verify()
            output_readable = True
        except Exception:
            output_readable = False

    # 6. Formatted print output
    print("GRAD-CAM TEST")
    print("-------------")
    print(f"Model loaded: {'YES' if model_loaded else 'NO'}")
    print(f"Test image: {test_img_path.name}")
    print(f"Predicted class: {predicted_class}")
    print(f"Grad-CAM generated: {'YES' if gradcam_generated else 'NO'}")
    print(f"Output image: {output_path}")
    print(f"Output image readable: {'YES' if output_readable else 'NO'}")

    if model_loaded and gradcam_generated and output_readable:
        print("GRAD-CAM TEST: PASSED")
    else:
        print("GRAD-CAM TEST: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
