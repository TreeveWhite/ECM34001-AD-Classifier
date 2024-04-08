# Tools

The files in this directory contains the python scripts which were used by the project. Some of these scripts were used multiple times and alteratives were only used once.

## delete_files.py

This file contains a python script to delete a certain number of images from a directory using a modulo counter to ensure a fair distribution of images being deleted from each MRI scan. This method was used to mitigate the risk of bias being introduced.

## plot_graphs.py

This file contains several procedured used to plot charts using MatPlotLib and Seaborne. Including plotting line graphs for traiining metrics, bar charts for testing metrics, and heatmaps for confusion matricies.

## reorganise_dataset.py

This file contains a python script to separate a dataset into separate training and testing datasets using a stratified sampling approach.

## training_analyser.py

This file contains a python script which uses regex to extract all the desired metrics from the training log of a deep learning model and organise the metrics into an excel file.
