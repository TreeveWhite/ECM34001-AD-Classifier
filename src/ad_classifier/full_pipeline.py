from ad_classifier.preprocessing.input_to_npy import extract_3dimg
from ad_classifier.preprocessing.npy_to_slice import extract_brain, get_slices, slice_to_img

from ad_classifier.postprocessing.attention_maps import get_attention_map


class FullPipeline:
    def __init__(self, slice_model, ad_model) -> None:
        self.slice_model = slice_model
        self.ad_model = ad_model

    def extract_slices(self, dcm_files):
        slices3d = extract_3dimg(dcm_files)

        good_slices_indexes = get_slices(slices3d)

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
        predictions = self.ad_model.predict(predictor_slices)

        # Average the class predictions of all predictor_slices
        avg_prediction = [
            sum([pred[i] for pred in predictions])/len(predictions) for i in range(4)]

        # Attention Maps
        attention_maps = [get_attention_map(
            img, self.ad_model) for img in predictor_slices]

        return avg_prediction, attention_maps
