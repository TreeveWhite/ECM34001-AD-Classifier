import os
import pydicom as dicom
import numpy as np
import pandas as pd

ADNI_DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI"
DATA_RESULTS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_3D"

METADATA_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/MPRAGE__CN_MCI_pMCI_AD__1_20_2024.csv"


def load_metadata(csv_path, columns=["Subject", "Group"]):
    df = pd.read_csv(csv_path, index_col="Image Data ID")
    df.replace({"EMCI": "MCI", "SMC": "MCI", "LMCI": "pMCI"}, inplace=True)
    return df[columns]


def make_classes_folders(parent_dir, classes):
    for _class in classes:
        group_path = os.path.join(
            parent_dir, _class)
        os.makedirs(group_path, exist_ok=True)


def load_dicom_series(dicom_files):
    dicom_files.sort()
    slices = [dicom.read_file(file) for file in dicom_files]
    return slices


def filter_order_slices(slices):
    resp = []
    skipcount = 0
    for s in slices:
        if hasattr(s, "SliceLocation"):
            resp.append(s)
        else:
            skipcount = skipcount + 1

    if skipcount != 0:
        print("WARNING: Skiped Slices with no SliceLocation: {}".format(skipcount))
    return sorted(resp, key=lambda s: s.SliceLocation)


def extract_3dimg(dcm_files):
    if len(dcm_files) < 5:
        print(f"WARNING: Not enough dcm slices")
        return None

    existing_slices = filter_order_slices(
        load_dicom_series(dcm_files))

    slice_position = existing_slices[0].ImageOrientationPatient
    img_shape = list(existing_slices[0].pixel_array.shape)
    img_shape.append(len(existing_slices))

    img3d = np.zeros(img_shape)
    for i, s in enumerate(existing_slices):
        img2d = s.pixel_array

        if slice_position == [0, 1, 0, 0, 0, -1]:
            # Input Slices are Sattigal
            img3d[:, :, i] = img2d
        elif slice_position == [1, 0, 0, 0, 0, -1]:
            # Input Slices are Coronal
            img3d[:, i, :] = img2d
        elif slice_position == [1, 0, 0, 0, 1, 0]:
            # Input Slices are Axial
            img3d[i, :, :] = img2d
    return img3d


def dataset_to_3dimg(metadata, dataset_path, results_path):
    # Create new folders in Results Path for the Classes
    make_classes_folders(results_path, metadata["Group"].unique())

    for root, _, files in os.walk(dataset_path):
        image_id = os.path.basename(root)
        if image_id in metadata.index:
            try:
                class_results_path = os.path.join(
                    results_path, metadata.at[image_id, "Group"])

                dcm_files = [os.path.join(root, file)
                             for file in files if file.endswith('.dcm')]

                img3d = extract_3dimg(dcm_files)

                save_path = os.path.join(class_results_path, image_id)
                np.save(f"{save_path}.npy", img3d)
            except Exception as e:
                print(f"Failed: {e}")


if __name__ == "__main__":

    metadata = load_metadata(METADATA_PATH)

    os.makedirs(DATA_RESULTS_PATH, exist_ok=True)

    dataset_to_3dimg(metadata, dataset_path=ADNI_DATASET_PATH,
                     results_path=DATA_RESULTS_PATH)
