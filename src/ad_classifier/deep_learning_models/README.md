# Model Training Files

This directory contains all the files used to train the deep learning models of the project on Google Colab. All files in this directory are Jupyter Notebooks.

## ad_model.pynb

This notebook contains the code that created and trained the five diagnostic deep learning models used by CogniCheck.

Additionally, this file was used to develop and train the 14 experimental deep learning models, which were utilized during the experimental process to develop CogniNet.

## slice_model.ipynb

This notebook contains the code used to train the axial slice relevance model. A transfer learning approach was employed, with an Inception V3 model pre-trained on the ImageNet dataset. Fine-tuning training was then performed on a smaller dataset of relevant and irrelevant axial slices.

## mdel_testing.py

This Python script contains the logic to test all of the diagnostic models in a directory. It loads in a testing dataset and logs all the testing metrics of each model's performance against it for every model in a directory.
