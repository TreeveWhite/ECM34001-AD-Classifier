"""
delete_files.py
==============================================
This file contains a small script used to rmove files from a directory using a
modulo counting aporach. This specific approach was used to ensure that when
remove axial slices from a dataset, the risk of an entire MRi can being removed
from the dataset is minimal and instead slices shouldbe removed proprotionally
from every MRI scan.
"""
import os


def delete_files(directory):
    """
    Itterate through dataset removing files.
    """
    files = os.listdir(directory)
    files.sort()
    count = 0

    for file in files:
        count += 1
        if count % 5 != 1:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


if __name__ == "__main__":
    directory_path = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_SLICE_TRAINING_DATA/BAD"
    delete_files(directory_path)
