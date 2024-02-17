import os
import sys
import cv2
from matplotlib import pyplot as plt
from matplotlib.image import imsave
import numpy as np
import pandas as pd
import tensorflow as tf

from exceptions import NoGoodSlicesException

METADATA_PATH = "/mnt/s/MPRAGE__CN_MCI_pMCI_AD__1_20_2024.csv"

DATA_RESULTS_PATH = "/mnt/s/ADNI_POST_PROCESS_3D"
SLICE_RESULTS_PATH = "/mnt/s/ADNI_POST_PROCESS_MODELED_SLICE"

SLICE_MODEL_PATH = "/mnt/s/slice_extraction_model.h5"

IMAGE_SIZE = [200, 200]


def load_metadata(csv_path, columns=["Subject", "Group"]):
    df = pd.read_csv(csv_path, index_col="Image Data ID")
    df.replace({"EMCI": "MCI", "SMC": "MCI", "LMCI": "pMCI"}, inplace=True)
    return df[columns]


def show_image(title, img, ctype, show):
    if show:
        plt.figure(figsize=(10, 10))
        if ctype == 'bgr':
            b, g, r = cv2.split(img)       # get b,g,r
            rgb_img = cv2.merge([r, g, b])     # switch it to rgb
            plt.imshow(rgb_img)
        elif ctype == 'hsv':
            rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
        elif ctype == 'grey':
            plt.imshow(img, cmap='grey')
        elif ctype == 'rgb':
            plt.imshow(img)
        else:
            raise Exception("Unknown colour type")
        plt.axis('off')
        plt.title(title)
        plt.show()


def make_classes_folders(parent_dir, classes):
    for _class in classes:
        group_path = os.path.join(
            parent_dir, _class)
        os.makedirs(group_path, exist_ok=True)


def extract_brain(img_slice, denoise=True):
    # Apply Otsu's automatic thresholding
    ret, thresh = cv2.threshold(
        img_slice, 0, 255, cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(thresh)

    marker_area = [np.sum(markers == m)
                   for m in range(np.max(markers)+1) if m != 0]

    largest_component = np.argmax(marker_area)+1
    brain_mask = markers == largest_component

    brain_out = img_slice.copy()

    brain_out[brain_mask == False] = 0

    if denoise:
        brain_out = cv2.GaussianBlur(brain_out, (3, 3), 0)

    return brain_out


def slice_to_img(data):
    img_slice_normalized = (
        data - np.min(data)) / (np.max(data) - np.min(data))
    img_slice_uint8 = (
        img_slice_normalized * 255).astype("uint8")
    image_slice_resized = cv2.resize(img_slice_uint8, (200, 200))
    return image_slice_resized


def get_slices(img3d):
    # Define Desired Slice Indexes
    slice_indexes = range(
        round(img3d.shape[0]*0.2), round(img3d.shape[0]*0.8))

    slices_dataset = []
    for index in slice_indexes:
        slice = img3d[index, :, :].T

        img_slice = slice_to_img(slice)

        image_data = cv2.cvtColor(
            img_slice, cv2.COLOR_BayerGB2BGR)

        slices_dataset.append(image_data)

    slice_images_array = np.array(slices_dataset)
    pred_scores = list(map(lambda x : x[0], SLICE_MODEL.predict(slice_images_array)))

    good_slices = [slice_indexes[i] for i, score in enumerate(pred_scores) if score == 1]

    if len(good_slices) < 3:
        sorted_indicies = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        next_best = [slice_indexes[i] for i in sorted_indicies if slice_indexes[i] not in good_slices and pred_scores[i] > 0.5]
        good_slices = good_slices + next_best[:3 - len(good_slices)]

    if len(good_slices) < 1:
        raise NoGoodSlicesException("No slices predicted >0.5")

    return good_slices


def npy_dataset_to_slice(data_path, results_path, denoise, show):
    for root, _, files in os.walk(data_path):
        class_name = os.path.basename(root)

        npy_files = [file for file in files if file.endswith(".npy")]

        for npy_file in npy_files:

            scan_id = npy_file.split('.')[0]

            print(scan_id)

            save_path = os.path.join(os.path.join(
                results_path, class_name), f"{scan_id}")

            img3d = np.load(os.path.join(root, npy_file))

            try:
                good_slices_indexes = get_slices(img3d)
            except NoGoodSlicesException as e:
                print(f"Unable to extract slices for {scan_id}")

            for index in good_slices_indexes:
                slice = img3d[index, :, :].T
                show_image("Preproessed", slice, "grey", show)

                # Preprocess -> Skull Extraction -> Guassian Denoise
                img_slice = slice_to_img(slice)
                post_processed_slice = extract_brain(extract_brain(
                    img_slice, False), denoise)
                show_image("Postprocessed", post_processed_slice, "grey", show)

                imsave(f"{save_path}-{index}.png",
                       post_processed_slice, cmap="grey")


if __name__ == "__main__":

    SLICE_MODEL = tf.keras.models.load_model(SLICE_MODEL_PATH)

    args = sys.argv[1:]

    os.makedirs(SLICE_RESULTS_PATH, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)

    classes = metadata["Group"].unique()
    make_classes_folders(SLICE_RESULTS_PATH, classes)

    npy_dataset_to_slice(data_path=DATA_RESULTS_PATH,
                         results_path=SLICE_RESULTS_PATH,
                         denoise=False if "no_denoise" in args else True,
                         show=True if "show" in args else False)
