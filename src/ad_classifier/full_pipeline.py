"""
full_pipeline.py
==============================================
THis file contains the FullPipeline object which is used to combine all the
procedures required to make a diagnosis together to enable easy interaction with
the deep learning models.
"""
from ad_classifier.preprocessing.input_to_npy import extract_3dimg
from ad_classifier.preprocessing.npy_to_slice import extract_brain, get_slices, slice_to_img

from ad_classifier.postprocessing.attention_maps import get_attention_map

import numpy as np
import keras
import cv2

CLASS_NAMES = ["AD", "CN", "MCI", "pMCI"]


class FullPipeline:
    """
    Full Pipeline
    """

    def __init__(self, slice_model_path, ad_model_path) -> None:
        """
        Creates a Full Pipeline object which loads in a specific AD DL model
        and a slice relevence model.
        """
        self.load_slice_model(slice_model_path)
        self.load_ad_model(ad_model_path)

    def load_slice_model(self, model_path):
        """
        Used to change the underlying model used to collect relevent
        axial slices.
        """
        self.slice_model = keras.models.load_model(model_path)

    def load_ad_model(self, model_path):
        """
        Used to change the underlyig diagnostic model used to create
        diagnoses.
        """
        self.ad_model = keras.models.load_model(model_path)

    def load_in_scan(self, img_path):
        """
        Used to load an axial slice from its path into the correct format to be used
        by a deep learning model (opens the image, reas it as greyscale and reshapes it)
        """
        return np.expand_dims(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), -1)

    def extract_slices(self, dcm_files):
        """
        This procedure is used to take in a complete MRI scan and then
        return the relevent axial slices.
        """
        # Convert MRI scan to Axial Slices
        slices3d = extract_3dimg(dcm_files)

        # Select the Revent Axial Slices
        good_slices_indexes = get_slices(slices3d, self.slice_model)

        predictor_slices = []
        for index in good_slices_indexes:
            slice = slices3d[index, :, :].T

            # Preprocess -> Skull Extraction -> Guassian Denoise
            img_slice = slice_to_img(slice)
            post_processed_slice = extract_brain(extract_brain(
                img_slice, False), True)

            predictor_slices.append(post_processed_slice)

        return predictor_slices

    def make_prediction(self, predictor_slices):
        """
        This procedure take in a list of axial slices and then uses them
        to make a diagnosis. This procedure returns a diagnosis and the
        attention maps for each axial slice used.
        """
        # Get predictions from AD model
        predictor_slices = np.array(predictor_slices)
        predictions = self.ad_model.predict(predictor_slices)

        # Average the class predictions of all predictor_slices
        avg_prediction = np.array([
            sum([pred[i] for pred in predictions])/len(predictions) for i in range(4)])

        diagnosis = CLASS_NAMES[avg_prediction.argmax(axis=-1)]

        # Get Attention Maps
        attention_maps = [get_attention_map(
            img, self.ad_model) for img in predictor_slices]

        return diagnosis, avg_prediction, attention_maps
