"""
attention_maps.py
==============================================
This file contains all the logic required to generate the attention maps of any
given deep learning model. The program uses a Grad-CAM implimentation and is programmed
in such a way that it can dynamically extract the final convolutional layer from
any of the deep learning model's implimented to enable the same procedures to work
with any model without needing to change any logic.
"""
import tensorflow as tf
import cv2
import numpy as np


def get_activations_at(input_image, model):
    """
    Extracts the final convolutional layerof a given model, and uses a new model
    to extract the activations at that layer for the given input image.
    """
    final_activation_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            final_activation_layer = layer
            break
    model = tf.keras.models.Model(
        inputs=model.inputs, outputs=final_activation_layer.output)
    return model.predict(np.expand_dims(
        input_image, 0))


def postprocess_activations(activations):
    """
    Resised and Converts the activations so they can be mapped ontop of the input
    image as a heatmp.
    """
    output = np.abs(activations)
    output = np.sum(output, axis=-1).squeeze()

    # resize and convert to image
    output = cv2.resize(output, (200, 200))
    output /= output.max()
    output *= 255
    return 255 - output.astype('uint8')


def apply_heatmap(weights, img):
    """
    Creates a heatmap based on the weights of the activations and layers itself
    ontop of the original input image.
    """
    # generate heat maps
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap


def get_attention_map(input_image, model):
    """
    Wrapper procedure to combine all the required procedures the create an
    attention map of a given input image using a given deep learning model.
    """
    activations = get_activations_at(input_image, model)
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, input_image)

    return heatmap
