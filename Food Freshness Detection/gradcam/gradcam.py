"""
gradcam/gradcam.py
------------------
Real Grad-CAM implementation for EfficientNet-B0 Food Freshness Detection model.

Uses GradientTape to compute gradients of the predicted class score with respect
to the activations of the final convolutional layer (top_activation).

All computations are performed on the actual model - no fake heatmaps, no placeholders.
"""

import io
import os
import uuid
from pathlib import Path

# Suppress TF info/warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 224

# Verified final activation layer of EfficientNet-B0 before GlobalAveragePooling:
# "top_activation" (shape: 7x7x1280)
GRADCAM_LAYER_NAME = "top_activation"


# ---------------------------------------------------------------------------
# Preprocessing (identical to training pipeline)
# ---------------------------------------------------------------------------

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes identically to the training pipeline:
    - RGB conversion
    - Resize to (IMG_SIZE, IMG_SIZE)
    - float32 array in [0, 255]
    - Add batch dimension -> (1, 224, 224, 3)
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)


def preprocess_image_path(image_path) -> np.ndarray:
    """Preprocess from a file path."""
    with open(image_path, "rb") as f:
        return preprocess_image_bytes(f.read())


# ---------------------------------------------------------------------------
# Grad-CAM Core
# ---------------------------------------------------------------------------

def build_gradcam_model(model: keras.Model, layer_name: str = GRADCAM_LAYER_NAME) -> keras.Model:
    """
    Build a sub-model that outputs both:
      - the activations of the target convolutional layer
      - the final class predictions
    """
    target_layer = model.get_layer(layer_name)
    model_output = model.output[0] if isinstance(model.output, list) else model.output
    gradcam_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model_output],
        name="gradcam_submodel",
    )
    return gradcam_model


def compute_gradcam(
    gradcam_model: keras.Model,
    preprocessed_tensor: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for class_index using GradientTape.

    Steps:
    1. Forward pass to obtain conv activations and class predictions.
    2. Compute gradients of the target class score w.r.t. conv feature maps.
    3. Global average pooling of gradients -> per-channel weights.
    4. Compute weighted sum of feature maps.
    5. Apply ReLU to retain only positive influence.
    6. Normalize heatmap safely to [0, 1].

    Returns:
        heatmap: float32 array of shape (7, 7) in [0, 1].
    """
    input_tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(input_tensor, training=False)
        if isinstance(predictions, list):
            predictions = predictions[0]
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Failed to compute gradients for Grad-CAM.")

    # Global average pooling across spatial dimensions (height, width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weight each feature map channel by its gradient importance
    # conv_outputs[0] shape: (H, W, C)
    weighted_conv = conv_outputs[0] * pooled_grads  # shape: (H, W, C)

    # Sum along channel dimension
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)  # shape: (H, W)

    # Apply ReLU: only positive contributions
    heatmap = tf.nn.relu(heatmap).numpy()

    # Safely normalize to [0, 1]
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay & File Generation
# ---------------------------------------------------------------------------

def create_heatmap_overlay(
    original_image_bytes: bytes,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap with the original image at its native resolution.

    Returns:
        overlay: uint8 RGB numpy array of shape (orig_H, orig_W, 3).
    """
    with Image.open(io.BytesIO(original_image_bytes)) as img:
        img_rgb = img.convert("RGB")
        orig_w, orig_h = img_rgb.size
        orig_array = np.array(img_rgb, dtype=np.float32) / 255.0

    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply color map
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # RGB channels

    # Alpha blending: overlay = alpha * heatmap + (1 - alpha) * original
    overlay = alpha * colored_heatmap + (1.0 - alpha) * orig_array
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def save_gradcam_image(
    overlay_array: np.ndarray,
    output_dir,
) -> str:
    """
    Save the Grad-CAM overlay image to output_dir with a unique filename.

    Returns:
        Absolute path to the saved image as a string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex[:12]
    filename = f"gradcam_{unique_id}.jpg"
    save_path = output_dir / filename

    pil_img = Image.fromarray(overlay_array, mode="RGB")
    pil_img.save(str(save_path), format="JPEG", quality=90)

    return str(save_path)


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: keras.Model,
    image_bytes: bytes,
    class_index: int,
    output_dir,
    layer_name: str = GRADCAM_LAYER_NAME,
) -> str:
    """
    Generate and save a real Grad-CAM overlay for the given predicted class index.

    Returns:
        Absolute path to the saved Grad-CAM image file.
    """
    preprocessed = preprocess_image_bytes(image_bytes)
    gradcam_model = build_gradcam_model(model, layer_name=layer_name)
    heatmap = compute_gradcam(gradcam_model, preprocessed, class_index)
    overlay = create_heatmap_overlay(image_bytes, heatmap)
    save_path = save_gradcam_image(overlay, output_dir)
    return save_path
