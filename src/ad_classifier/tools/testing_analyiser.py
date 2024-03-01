import os
import sys
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

TEST_DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_TEST_DATASET"
IMAGE_SIZE = [200, 200]


def test_model(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    sensitivity = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity (Recall):", sensitivity)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(conf_matrix)


def test_models_in_dir(directory, test_images, y_true):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in {"OLD MODELS"}]
        for file in files:
            if file.endswith(".h5"):
                print("======================================")
                print("TESTING: ", os.path.basename(root))
                model = keras.models.load_model(os.path.join(root, file))

                y_pred = np.argmax(model.predict(test_images), axis=1)

                test_model(y_pred, y_true)


if __name__ == "__main__":
    models_path = sys.argv[-1]

    test_ds = image_dataset_from_directory(
        TEST_DATASET_PATH,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        shuffle=False,
    ).prefetch(buffer_size=AUTOTUNE)

    y_true = []
    test_images = [image for image, _ in test_ds]
    test_images = tf.concat(test_images, axis=0)

    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    test_models_in_dir(models_path, test_images, y_true)
