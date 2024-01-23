import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.image import imsave
import pydicom as dicom
import numpy as np
import pandas as pd

DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI"
DATASET_METADATA_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/MPRAGE__CN_MCI_pMCI_AD__1_20_2024.csv"
RESULTS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS"


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


def ShowImage(title, img, ctype, show=False):
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
    # apply Otsu's automatic thresholding
    ret, thresh = cv2.threshold(
        img_slice, 0, 255, cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(thresh)

    marker_area = [np.sum(markers == m)
                   for m in range(np.max(markers)) if m != 0]

    largest_component = np.argmax(marker_area)+1
    brain_mask = markers == largest_component

    brain_out = img_slice.copy()

    brain_out[brain_mask == False] = 0

    return brain_out


if __name__ == "__main__":

    os.makedirs(RESULTS_PATH, exist_ok=True)

    metadata = load_metadata(DATASET_METADATA_PATH)

    for root, dirs, files in os.walk(DATASET_PATH):
        current_folder = os.path.basename(root)
        if current_folder in metadata.index:
            group_path = os.path.join(
                RESULTS_PATH, metadata.at[current_folder, "Group"])
            os.makedirs(group_path, exist_ok=True)
            satigal_slices = filter_order_slices(
                load_dicom_series([os.path.join(root, file) for file in files if file.endswith('.dcm')]))

            ps = satigal_slices[0].PixelSpacing
            ss = satigal_slices[0].SliceThickness
            ax_aspect = ps[1] / ps[0]

            img_shape = list(satigal_slices[0].pixel_array.shape)
            img_shape.append(len(satigal_slices))
            img3d = np.zeros(img_shape)

            for i, s in enumerate(satigal_slices):
                img2d = s.pixel_array
                img3d[:, :, i] = img2d

            save_path = os.path.join(group_path, current_folder)
            np.save(f"{save_path}.npy", img3d)

            slice_indices = range(100, 105)

            for i, index in enumerate(slice_indices):
                img_slice = img3d[index, :, :].T

                ShowImage(
                    f"Pre Prepropressing Axial Slice {i}", img_slice, "grey")

                img_normalized = cv2.normalize(img_slice, None, 0, 1.0,
                                               cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                img_slice_uint8 = img_slice.astype("uint8")

                ShowImage(
                    f"Normalised Axial Slice {i}", img_slice_uint8, "grey")

                brain_out = extract_brain(img_slice_uint8)

                ShowImage(f"Brain Out Slice {i}", brain_out, 'grey')

                resized_brain_out = cv2.resize(
                    brain_out, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                ShowImage(
                    f"Resized Brain Out Slice {i}", resized_brain_out, 'grey')

                imsave(f"{save_path}-{i}.png",
                       resized_brain_out, cmap="grey")
