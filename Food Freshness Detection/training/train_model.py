"""
train_model.py
--------------
Transfer-learning with EfficientNet-B0 (ImageNet weights) for the
Food Freshness Detection dataset (20 classes).

Two-stage training
  Stage 1 : Train classification head only  (base frozen)
  Stage 2 : Fine-tune top layers of base    (small LR)

Saves
  model/food_freshness_efficientnet_b0.keras   – best checkpoint
  model/class_names.json                        – class-index mapping
  training/results/training_history.json        – full epoch history
  training/results/training_curves.png          – loss/accuracy curves

Usage
  python training/train_model.py
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress TF info/warning logs before importing TF
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe on Windows
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent           # .../training/
PROJECT_ROOT = SCRIPT_DIR.parent                         # .../Food_Freshness_Detection/

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "validation"
TEST_DIR    = DATASET_DIR / "test"

MODEL_DIR   = PROJECT_ROOT / "model"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_PATH        = MODEL_DIR / "food_freshness_efficientnet_b0.keras"
CLASS_NAMES_PATH  = MODEL_DIR / "class_names.json"
HISTORY_PATH      = RESULTS_DIR / "training_history.json"
CURVES_PATH       = RESULTS_DIR / "training_curves.png"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE    = 224          # EfficientNetB0 native resolution
BATCH_SIZE  = 16           # conservative for CPU / modest RAM
SEED        = 42

# Stage-1 head training
HEAD_EPOCHS = 20
HEAD_LR     = 1e-3

# Stage-2 fine-tuning
FINETUNE_EPOCHS    = 30
FINETUNE_LR        = 1e-5
FINETUNE_FROM_LAYER = 200   # unfreeze layers from this index onward

EXPECTED_CLASSES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_directories():
    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "validation"), (TEST_DIR, "test")]:
        if not d.is_dir():
            print(f"[ERROR] {name} directory not found: {d}")
            sys.exit(1)
        classes = [c for c in d.iterdir() if c.is_dir()]
        if len(classes) == 0:
            print(f"[ERROR] No class folders found in {d}")
            sys.exit(1)
        print(f"  {name:<12}: {d}  ({len(classes)} classes)")


def build_datasets():
    """
    Create tf.data pipelines for train / val / test.
    EfficientNetB0 expects pixel values in [0, 255] – the Keras application
    applies its own internal rescaling.  We must NOT add an extra Rescaling
    layer or divide by 255 manually.
    """

    # --- augmentation applied ONLY to training images ---
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="augmentation")

    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32)   # keep [0,255]; EfficientNet handles scaling
        image = augmentation(image, training=True)
        return image, label

    def preprocess_eval(image, label):
        image = tf.cast(image, tf.float32)
        return image, label

    common_kwargs = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        seed=SEED,
    )

    train_ds_raw = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        **common_kwargs,
    )

    val_ds_raw = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        shuffle=False,
        **common_kwargs,
    )

    # Capture class names from training split (alphabetically sorted by Keras)
    class_names = train_ds_raw.class_names
    n_classes   = len(class_names)
    print(f"\n  Discovered {n_classes} classes: {class_names}")

    if n_classes != EXPECTED_CLASSES:
        print(f"[ERROR] Expected {EXPECTED_CLASSES} classes, found {n_classes}.")
        sys.exit(1)

    # Validate val/test class names match training
    val_classes  = val_ds_raw.class_names
    if val_classes != class_names:
        print("[ERROR] Validation class names do not match training class names.")
        print(f"  Train: {class_names}")
        print(f"  Val  : {val_classes}")
        sys.exit(1)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds_raw
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names


def build_model(n_classes: int) -> keras.Model:
    """Build EfficientNetB0 transfer-learning model."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # EfficientNetB0 with its own built-in preprocessing
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None,
    )
    base_model.trainable = False     # freeze for Stage 1

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="bn_top")(x)
    x = layers.Dropout(0.3, name="dropout_top")(x)
    x = layers.Dense(256, activation="relu", name="fc_256")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Dropout(0.3, name="dropout_fc")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FoodFreshness_EfficientNetB0")
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def save_curves(history_stage1, history_stage2):
    """Plot and save combined training curves."""
    acc1  = history_stage1.history["accuracy"]
    vacc1 = history_stage1.history["val_accuracy"]
    loss1 = history_stage1.history["loss"]
    vloss1= history_stage1.history["val_loss"]

    acc2  = history_stage2.history.get("accuracy", [])
    vacc2 = history_stage2.history.get("val_accuracy", [])
    loss2 = history_stage2.history.get("loss", [])
    vloss2= history_stage2.history.get("val_loss", [])

    acc   = acc1  + acc2
    vacc  = vacc1 + vacc2
    loss  = loss1 + loss2
    vloss = vloss1+ vloss2
    epochs = range(1, len(acc) + 1)
    split  = len(acc1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EfficientNetB0 – Training History", fontsize=14)

    # Accuracy
    axes[0].plot(epochs, acc,  "b-",  label="Train Acc")
    axes[0].plot(epochs, vacc, "b--", label="Val Acc")
    if split < len(acc):
        axes[0].axvline(x=split + 0.5, color="gray", linestyle=":", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, loss,  "r-",  label="Train Loss")
    axes[1].plot(epochs, vloss, "r--", label="Val Loss")
    if split < len(loss):
        axes[1].axvline(x=split + 0.5, color="gray", linestyle=":", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CURVES_PATH, dpi=120)
    plt.close()
    print(f"  Curves saved -> {CURVES_PATH}")


def model_reload_test(class_names):
    """Load saved model, run one real prediction, confirm it works."""
    print("\n--- Model Reload Test ---")
    import random
    from pathlib import Path as P

    loaded_model = keras.models.load_model(str(MODEL_PATH))
    print("  Model loaded from disk successfully.")

    # Pick a random real test image
    test_root = TEST_DIR
    all_images = list(test_root.rglob("*.jpg")) + list(test_root.rglob("*.jpeg")) + list(test_root.rglob("*.png"))
    if not all_images:
        print("  [WARNING] No test images found for reload test.")
        return "FAILED"

    rng = random.Random(SEED)
    sample = rng.choice(all_images)
    actual_class = sample.parent.name

    img = keras.utils.load_img(str(sample), target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
    img_array = keras.utils.img_to_array(img)           # shape (224,224,3), values [0,255]
    img_array = np.expand_dims(img_array, axis=0)       # (1,224,224,3)

    preds = loaded_model.predict(img_array, verbose=0)
    pred_idx   = int(np.argmax(preds[0]))
    pred_class = class_names[pred_idx]
    confidence = float(preds[0][pred_idx])

    print(f"  Sample image : {sample.name}")
    print(f"  Actual class : {actual_class}")
    print(f"  Predicted    : {pred_class}")
    print(f"  Confidence   : {confidence:.4f}")

    return "PASSED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EFFICIENTNET-B0 TRAINING")
    print(f"  TensorFlow {tf.__version__}  |  Batch={BATCH_SIZE}  Seed={SEED}")
    print("=" * 60)

    # ---- Setup ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Checking dataset directories ...")
    check_directories()

    print("\n[2] Building tf.data pipelines ...")
    train_ds, val_ds, class_names = build_datasets()

    train_count = sum(1 for _ in TRAIN_DIR.rglob("*.jpg")) + \
                  sum(1 for _ in TRAIN_DIR.rglob("*.jpeg")) + \
                  sum(1 for _ in TRAIN_DIR.rglob("*.png"))
    val_count   = sum(1 for _ in VAL_DIR.rglob("*.jpg")) + \
                  sum(1 for _ in VAL_DIR.rglob("*.jpeg")) + \
                  sum(1 for _ in VAL_DIR.rglob("*.png"))

    print(f"  Train images      : {train_count}")
    print(f"  Validation images : {val_count}")

    # ---- Save class names ----
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    print(f"  Class mapping saved -> {CLASS_NAMES_PATH}")

    # ---- Build model ----
    print("\n[3] Building model ...")
    model, base_model = build_model(len(class_names))
    model.summary(print_fn=lambda s: print("  " + s))

    # ---- STAGE 1: Train head only ----
    print(f"\n[4] Stage 1 – Training classification head (base frozen, LR={HEAD_LR}) ...")
    compile_model(model, HEAD_LR)

    callbacks_stage1 = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    best_val_acc_stage1 = max(history_stage1.history["val_accuracy"])
    print(f"\n  Stage 1 best val accuracy: {best_val_acc_stage1:.4f}")

    # ---- STAGE 2: Fine-tune top layers ----
    print(f"\n[5] Stage 2 – Fine-tuning from layer {FINETUNE_FROM_LAYER} (LR={FINETUNE_LR}) ...")
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[FINETUNE_FROM_LAYER:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    n_trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  Trainable layers after unfreezing: {n_trainable}")

    compile_model(model, FINETUNE_LR)

    callbacks_stage2 = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
    ]

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    best_val_acc_stage2 = max(history_stage2.history["val_accuracy"])
    best_val_acc_overall = max(best_val_acc_stage1, best_val_acc_stage2)
    print(f"\n  Stage 2 best val accuracy : {best_val_acc_stage2:.4f}")
    print(f"  Overall best val accuracy : {best_val_acc_overall:.4f}")

    # ---- Save training history ----
    print("\n[6] Saving training history ...")
    combined_history = {
        "stage1": {k: [float(v) for v in vals]
                   for k, vals in history_stage1.history.items()},
        "stage2": {k: [float(v) for v in vals]
                   for k, vals in history_stage2.history.items()},
        "best_val_accuracy_stage1":  float(best_val_acc_stage1),
        "best_val_accuracy_stage2":  float(best_val_acc_stage2),
        "best_val_accuracy_overall": float(best_val_acc_overall),
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f"  History saved -> {HISTORY_PATH}")

    # ---- Save training curves ----
    print("\n[7] Saving training curves ...")
    save_curves(history_stage1, history_stage2)

    # ---- Model reload test ----
    reload_status = model_reload_test(class_names)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("## MODEL TRAINING RESULT")
    print("=" * 60)
    print(f"Model              : EfficientNet-B0")
    print(f"Classes            : {len(class_names)}")
    print(f"Training images    : {train_count}")
    print(f"Validation images  : {val_count}")
    print(f"\nBest val accuracy  : {best_val_acc_overall:.4f}")
    print(f"\nModel saved        : {'YES' if MODEL_PATH.exists() else 'NO'}")
    print(f"Class mapping saved: {'YES' if CLASS_NAMES_PATH.exists() else 'NO'}")
    print(f"Prediction reload  : {reload_status}")
    print("=" * 60)
    print("\nTraining complete. Run evaluate_model.py for test metrics.\n")


if __name__ == "__main__":
    main()
