# PreProcessing

The files within this folder are used to perform the preprocessing stages of the project.

## data_augmentation.py

This file contains all the logic required to augment datasets up to a given number of images per class.

## exceptions.py

This file contains custom exceptions used by the project to handle errors.

## input_to_npy.py

This file contains the first stage of the preprocessing pipeline, which takes DICOM files and converts them into a 3D array representing the entire MRI scan. This 3D array is stored as a NumPy file.

## npy_to_slice.py

This file contains the second stage of the preprocessing stage, which employs a binary classification CNN model to determine which axial slices of the full MRI scan are relevant to making a diagnosis.
