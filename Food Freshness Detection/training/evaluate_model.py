"""
evaluate_model.py
-----------------
Evaluate the saved EfficientNet-B0 model on the test dataset.

Computes
  * Test loss & accuracy
  * Per-class precision, recall, F1-score
  * Confusion matrix

Saves
  training/results/confusion_matrix.png
  training/results/classification_report.json

Usage
  python training/evaluate_model.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TEST_DIR    = DATASET_DIR / "test"

MODEL_DIR        = PROJECT_ROOT / "model"
MODEL_PATH       = MODEL_DIR / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

RESULTS_DIR   = SCRIPT_DIR / "results"
CM_PATH       = RESULTS_DIR / "confusion_matrix.png"
REPORT_PATH   = RESULTS_DIR / "classification_report.json"

IMG_SIZE   = 224
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"[ERROR] class_names.json not found: {CLASS_NAMES_PATH}")
        print("  Run train_model.py first.")
        sys.exit(1)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_names"]


def load_model():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Saved model not found: {MODEL_PATH}")
        print("  Run train_model.py first.")
        sys.exit(1)
    print(f"  Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(str(MODEL_PATH))
    print("  Model loaded.")
    return model


def build_test_dataset(class_names):
    if not TEST_DIR.is_dir():
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        sys.exit(1)

    test_ds_raw = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
    )

    # Verify class order matches training
    if test_ds_raw.class_names != class_names:
        print("[ERROR] Test class names do not match class_names.json.")
        print(f"  Expected : {class_names}")
        print(f"  Found    : {test_ds_raw.class_names}")
        sys.exit(1)

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = (
        test_ds_raw
        .map(lambda img, lbl: (tf.cast(img, tf.float32), lbl),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    return test_ds, test_ds_raw


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a labelled confusion matrix."""
    n = len(class_names)
    fig_size = max(12, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Confusion Matrix – EfficientNetB0 (Test Set)", fontsize=13)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EFFICIENTNET-B0 EVALUATION")
    print(f"  TensorFlow {tf.__version__}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load class names ----
    print("\n[1] Loading class names ...")
    class_names = load_class_names()
    print(f"  {len(class_names)} classes: {class_names}")

    # ---- Load model ----
    print("\n[2] Loading model ...")
    model = load_model()

    # ---- Build test dataset ----
    print("\n[3] Building test dataset ...")
    test_ds, test_ds_raw = build_test_dataset(class_names)

    test_count = sum(1 for _ in TEST_DIR.rglob("*.jpg")) + \
                 sum(1 for _ in TEST_DIR.rglob("*.jpeg")) + \
                 sum(1 for _ in TEST_DIR.rglob("*.png"))
    print(f"  Test images : {test_count}")

    # ---- Evaluate ----
    print("\n[4] Evaluating model on test set ...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")

    # ---- Collect true and predicted labels ----
    print("\n[5] Collecting predictions for per-class metrics ...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- Classification report ----
    print("\n[6] Computing classification report ...")
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )
    print("\n" + report_str)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    # ---- Confusion matrix ----
    print("\n[7] Computing confusion matrix ...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, CM_PATH)

    # ---- Save JSON report ----
    print("\n[8] Saving classification report ...")
    full_report = {
        "test_loss":     float(test_loss),
        "test_accuracy": float(test_acc),
        "test_images":   int(test_count),
        "num_classes":   len(class_names),
        "class_names":   class_names,
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall":    round(report_dict[cls]["recall"],    4),
                "f1_score":  round(report_dict[cls]["f1-score"],  4),
                "support":   int(report_dict[cls]["support"]),
            }
            for cls in class_names
        },
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall":    round(report_dict["macro avg"]["recall"],    4),
            "f1_score":  round(report_dict["macro avg"]["f1-score"],  4),
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall":    round(report_dict["weighted avg"]["recall"],    4),
            "f1_score":  round(report_dict["weighted avg"]["f1-score"],  4),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"  Report saved -> {REPORT_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("## EVALUATION RESULT")
    print("=" * 60)
    print(f"Test images      : {test_count}")
    print(f"Test loss        : {test_loss:.4f}")
    print(f"Test accuracy    : {test_acc:.4f}")
    print(f"Macro F1         : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1      : {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"\nConfusion matrix : {'YES' if CM_PATH.exists() else 'NO'}")
    print(f"Class report     : {'YES' if REPORT_PATH.exists() else 'NO'}")
    print(f"\nEvaluation completed: YES")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
