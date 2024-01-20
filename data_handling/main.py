import os
from matplotlib.image import imsave
import pydicom as dicom
import numpy as np
import pandas as pd

DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI"
DATASET_METADATA_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/MPRAGE__CN_MCI_pMCI_AD__1_20_2024.csv"
RESULTS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS"


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

            # os.makedirs(current_folder, exist_ok=True)

            # slice_indices = np.linspace(0, img_shape[0] - 1, 6, dtype=int)

            # for i, index in enumerate(slice_indices):
            #     imsave(os.path.join(current_folder,
            #            f"slice_{i + 1}.png"), img3d[index, :, :].T, cmap="gray")

            slice_index = 102
            imsave(f"{save_path}.png",
                   img3d[slice_index, :, :].T, cmap="gray")
