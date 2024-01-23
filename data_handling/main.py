import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.image import imsave
import pydicom as dicom
import numpy as np
import imutils
import pandas as pd

ADNI_DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI"
METADATA_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/MPRAGE__CN_MCI_pMCI_AD__1_20_2024.csv"

DATA_RESULTS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_3D"
SLICE_RESULTS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_SLICE"

IMAGE_SIZE = [500, 500]


def load_metadata(csv_path, columns=["Subject", "Group"]):
    df = pd.read_csv(csv_path, index_col='Image Data ID')
    return df[columns]


def load_dicom_series(dicom_files):
    dicom_files.sort()
    slices = [dicom.read_file(file) for file in dicom_files]
    return slices


def filter_order_slices(slices):
    resp = []
    skipcount = 0
    for s in slices:
        if hasattr(s, 'SliceLocation'):
            resp.append(s)
        else:
            skipcount = skipcount + 1

    if skipcount != 0:
        print("WARNING: Skiped Slices with no SliceLocation: {}".format(skipcount))
    return sorted(resp, key=lambda s: s.SliceLocation)


def show_image(title, img, ctype, show=False):
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


def extract_brain(img_slice):
    # Apply Otsu's automatic thresholding
    ret, thresh = cv2.threshold(
        img_slice, 0, 255, cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(thresh)

    marker_area = [np.sum(markers == m)
                   for m in range(np.max(markers)) if m != 0]

    largest_component = np.argmax(marker_area)+1
    brain_mask = markers == largest_component

    brain_out = img_slice.copy()

    brain_out[brain_mask == False] = 0

    # img_denoised = cv2.GaussianBlur(brain_out, (3, 3), 0)

    return brain_out


def make_classes_folders(parent_dir, classes):
    for _class in classes:
        group_path = os.path.join(
            parent_dir, _class)
        os.makedirs(group_path, exist_ok=True)


def sagittal_to_3dimg(metadata, dataset_path, results_path):
    # Create new folders in Results Path for the Classes
    make_classes_folders(results_path, metadata["Group"].unique())

    for root, _, files in os.walk(dataset_path):
        image_id = os.path.basename(root)
        if image_id in metadata.index:
            class_results_path = os.path.join(
                results_path, metadata.at[image_id, "Group"])

            satigal_slices = filter_order_slices(
                load_dicom_series([os.path.join(root, file) for file in files if file.endswith('.dcm')]))

            img_shape = list(satigal_slices[0].pixel_array.shape)
            img_shape.append(len(satigal_slices))
            img3d = np.zeros(img_shape)
            for i, s in enumerate(satigal_slices):
                img2d = s.pixel_array
                img3d[:, :, i] = img2d

            save_path = os.path.join(class_results_path, image_id)
            np.save(f"{save_path}.npy", img3d)


def npy_to_slice(metadata, data_path, results_path):
    # Create new folders in Results Path for the Classes
    classes = metadata["Group"].unique()
    make_classes_folders(results_path, classes)

    for root, _, files in os.walk(data_path):
        class_name = os.path.basename(root)
        if class_name not in classes:
            continue
        npy_files = [file for file in files if file.endswith(".npy")]
        for npy_file in npy_files:
            save_path = os.path.join(os.path.join(
                results_path, class_name), f"{npy_file.split('.')[0]}.png")

            img3d = np.load(os.path.join(root, npy_file))

            slice_indices = range(100, 105)

            for i, index in enumerate(slice_indices):
                img_slice = img3d[index, :, :].T

                imsave(f"{save_path}-{i}-original.png",
                       img_slice, cmap="grey")

                # Normalise
                img_slice_uint8 = img_slice.astype("uint8")
                # Skull Extraction & Guassian Denoise
                brain_out = extract_brain(img_slice_uint8)
                # Resize (Aspect ratio Aware)
                resized_brain_out = imutils.resize(
                    brain_out, width=IMAGE_SIZE[0])

                imsave(f"{save_path}-{i}.png",
                       resized_brain_out, cmap="grey")


if __name__ == "__main__":

    os.makedirs(DATA_RESULTS_PATH, exist_ok=True)
    os.makedirs(SLICE_RESULTS_PATH, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)

    sagittal_to_3dimg(metadata, dataset_path=ADNI_DATASET_PATH,
                      results_path=DATA_RESULTS_PATH)

    npy_to_slice(metadata, data_path=DATA_RESULTS_PATH,
                 results_path=SLICE_RESULTS_PATH)
