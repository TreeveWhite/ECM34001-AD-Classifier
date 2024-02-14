import tensorflow as tf
import cv2
import numpy as np


def get_activations_at(input_image, model):
    final_activation_layer = None
    for layer in reversed(model.layers):
        if "batch_normalization" in layer.name:
            final_activation_layer = layer
            break
    model = tf.keras.models.Model(
        inputs=model.inputs, outputs=final_activation_layer.output)
    return model.predict(np.expand_dims(
        input_image, 0))


def postprocess_activations(activations):
    output = np.abs(activations)
    output = np.sum(output, axis=-1).squeeze()

    # resize and convert to image
    output = cv2.resize(output, (200, 200))
    output /= output.max()
    output *= 255
    return 255 - output.astype('uint8')


def apply_heatmap(weights, img):
    # generate heat maps
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap


def get_attention_map(input_image, model):
    activations = get_activations_at(input_image, model)
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, input_image)

    return heatmap
