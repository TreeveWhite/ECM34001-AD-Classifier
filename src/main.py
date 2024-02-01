import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
import keras
import tensorflow_addons as tfa

from config import IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES

DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_SLICE"


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn


if __name__ == "__main__":

    train_ds = image_dataset_from_directory(
        DATASET_PATH,
        labels="inferred",
        image_size=IMAGE_SIZE,
        color_mode="grayscale"
    )

    print((*IMAGE_SIZE, 3))

    print(f"Recognised Classes: {train_ds.class_names}")

    base_model = tf.keras.applications.DenseNet201(
        include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.models.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        tfa.layers.WeightNormalization(
            keras.layers.Dense(256, activation='relu')),
        keras.layers.Dropout(0.5),
        tfa.layers.WeightNormalization(keras.layers.Dense(
            len(train_ds.class_names), activation='softmax'))
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy(name='acc')]
    )

    history = model.fit(
        train_ds,
        epochs=10
    )
