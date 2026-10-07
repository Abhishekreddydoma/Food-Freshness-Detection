"""
predict_test.py
---------------
Verification script to reload the trained EfficientNet-B0 model,
load class mappings, preprocess a real test image identically to training,
run model prediction, derive produce type and freshness, and save the result.

Usage:
  python training/predict_test.py
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Suppress TF info/warning logs before importing TF
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent           # .../training/
PROJECT_ROOT = SCRIPT_DIR.parent                         # .../Food_Freshness_Detection/

MODEL_DIR        = PROJECT_ROOT / "model"
MODEL_PATH       = MODEL_DIR / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

DATASET_DIR = PROJECT_ROOT / "dataset"
TEST_DIR    = DATASET_DIR / "test"

RESULTS_DIR      = SCRIPT_DIR / "results"
OUTPUT_JSON_PATH = RESULTS_DIR / "prediction_test.json"

IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_mapping(path: Path) -> list:
    """Load class names from JSON mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Class names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "class_names" not in data or not isinstance(data["class_names"], list):
        raise ValueError(f"Invalid format in class names file: {path}")
    return data["class_names"]


def load_saved_model(path: Path) -> keras.Model:
    """Load the trained Keras model from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return keras.models.load_model(str(path))


def find_first_valid_test_image(test_dir: Path):
    """
    Locate one real valid image from dataset/test/.
    Returns (image_path, filename, actual_class_name).
    """
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Search through class subdirectories
    for class_folder in sorted(test_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        for img_file in sorted(class_folder.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    with Image.open(img_file) as img:
                        img.load()
                    return img_file, img_file.name, class_folder.name
                except Exception:
                    continue

    raise FileNotFoundError(f"No valid test images found in {test_dir}")


def preprocess_image(image_path: Path) -> tf.Tensor:
    """
    Preprocess a single image matching the training/eval pipeline:
    - Load RGB image resized to (IMG_SIZE, IMG_SIZE)
    - Convert to float32 tensor with pixel values in [0, 255]
    - Add batch dimension -> shape (1, IMG_SIZE, IMG_SIZE, 3)
    """
    img = keras.utils.load_img(
        str(image_path),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        interpolation="bilinear",
    )
    img_array = keras.utils.img_to_array(img)  # float32 array in [0, 255]
    img_tensor = tf.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    return img_tensor


def parse_produce_and_freshness(class_name: str):
    """
    Derive produce type and freshness condition from the class name.
    Examples:
      FreshApple -> ('Apple', 'Fresh')
      RottenApple -> ('Apple', 'Rotten')
    """
    if class_name.startswith("Fresh"):
        return class_name[len("Fresh"):], "Fresh"
    elif class_name.startswith("Rotten"):
        return class_name[len("Rotten"):], "Rotten"
    return class_name, "Unknown"


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    model_loaded = False
    class_mapping_loaded = False
    image_loaded = False
    preprocessing_succeeded = False
    prediction_succeeded = False

    try:
        # 1. Load saved model
        model = load_saved_model(MODEL_PATH)
        model_loaded = True

        # 2. Load class mapping
        class_names = load_class_mapping(CLASS_NAMES_PATH)
        class_mapping_loaded = True

        # 3. Find one real valid test image
        test_image_path, test_image_name, actual_class = find_first_valid_test_image(TEST_DIR)
        image_loaded = True

        # 4. Preprocess image matching training pipeline
        preprocessed_tensor = preprocess_image(test_image_path)
        preprocessing_succeeded = True

        # 5. Run real model prediction
        predictions = model.predict(preprocessed_tensor, verbose=0)
        probabilities = predictions[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        prediction_succeeded = True

        # 6. Derive produce type and freshness
        produce, freshness = parse_produce_and_freshness(predicted_class)

        # 7. Print formatted summary
        print("MODEL RELOAD TEST")
        print("-----------------")
        print(f"Model loaded: {'YES' if model_loaded else 'NO'}")
        print(f"Class mapping loaded: {'YES' if class_mapping_loaded else 'NO'}")
        print(f"Test image: {test_image_name}")
        print(f"Actual class: {actual_class}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Produce: {produce}")
        print(f"Freshness: {freshness}")
        print()

        # 8. Save result to JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "test_image": test_image_name,
            "image_path": str(test_image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "produce": produce,
            "freshness": freshness,
            "confidence": confidence,
            "model_loaded": model_loaded,
            "class_mapping_loaded": class_mapping_loaded,
            "status": "PASSED"
        }

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        # 9. Verify success condition and print status
        if (
            model_loaded
            and class_mapping_loaded
            and image_loaded
            and preprocessing_succeeded
            and prediction_succeeded
        ):
            print("MODEL RELOAD TEST: PASSED")
        else:
            print("MODEL RELOAD TEST: FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Prediction test failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
