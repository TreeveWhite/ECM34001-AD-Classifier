from ad_classifier.preprocessing.input_to_npy import extract_3dimg
from ad_classifier.preprocessing.npy_to_slice import extract_brain, get_slices, slice_to_img

from ad_classifier.postprocessing.attention_maps import get_attention_map

import numpy as np
import keras
import cv2


class FullPipeline:
    def __init__(self, slice_model_path, ad_model_path) -> None:
        self.load_slice_model(slice_model_path)
        self.load_ad_model(ad_model_path)

    def load_slice_model(self, model_path):
        self.slice_model = keras.models.load_model(model_path)

    def load_ad_model(self, model_path):
        self.ad_model = keras.models.load_model(model_path)

    def load_in_scan(self, img_path):
        return np.expand_dims(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), -1)

    def extract_slices(self, dcm_files):
        slices3d = extract_3dimg(dcm_files)

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
        # AD Model
        predictor_slices = np.array(predictor_slices)
        predictions = self.ad_model.predict(predictor_slices)

        # Average the class predictions of all predictor_slices
        avg_prediction = [
            sum([pred[i] for pred in predictions])/len(predictions) for i in range(4)]

        # Attention Maps
        attention_maps = [get_attention_map(
            img, self.ad_model) for img in predictor_slices]

        return avg_prediction, attention_maps
