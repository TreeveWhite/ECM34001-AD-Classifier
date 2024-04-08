# Model Training Files

This directory contains all the files which were used to train the deep learning models of the project on Google Colab. For that reason all of the files are Jupiter Notebooks.

## ad_model.pynb

This file contains the code which created and trained the five diagnostic deep learning models used by CogniCheck.

This same file was aso used to develop and train the 14 experimental deep learning models which were used during the experimental process to develop CogniNet.

## slice_model.ipynb

This file contains the code used to train the axial slice relevence model. A transfered learning approach was used with an Inception V3 model being pretrained on the ImageNet dataset before fine tune training was perfomed on a smaller dataset of relevent and irrelevent axial slices.

## mdel_testing.py

This file contains the logic to test all of te diagnostic models in a directory. It loads in a testing dataset and logs all the testing metrics of each model's performance against it for every model in a directory.
