import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("model/food_freshness_efficientnet_b0.keras")
target_layer = model.get_layer("top_activation")
model_output = model.output[0] if isinstance(model.output, list) else model.output
gradcam_model = keras.Model(inputs=model.inputs, outputs=[target_layer.output, model_output])

x = tf.convert_to_tensor(np.zeros((1, 224, 224, 3), dtype=np.float32))

with tf.GradientTape() as tape:
    conv_outputs, predictions = gradcam_model(x, training=False)
    if isinstance(predictions, list):
        predictions = predictions[0]
    class_score = predictions[:, 0]

grads = tape.gradient(class_score, conv_outputs)
print("conv_outputs shape:", conv_outputs.shape)
print("predictions shape:", predictions.shape)
print("grads shape:", grads.shape if grads is not None else None)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
print("pooled_grads shape:", pooled_grads.shape)
heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
heatmap = tf.nn.relu(heatmap)
print("heatmap shape:", heatmap.shape)
