import sys
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import InceptionV3
from datetime import datetime
import os

from config import IMAGE_SIZE, MODEL_SAVE_PATH, SLICE_EXTRACTION_DATASET_PATH

STRATEGY = tf.distribute.get_strategy()
AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync
EPOCHS = 10

MODEL_NAME = "SLICE_EXTRACTOR_DENSE_NET"


if __name__ == "__main__":
    base_dir = f"{MODEL_SAVE_PATH}/{MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    os.makedirs(base_dir, exist_ok=True)

    with open(f"{base_dir}/log.log", "w") as sys.stdout:

        print("REPLICAS: ", STRATEGY.num_replicas_in_sync)

        train_ds = image_dataset_from_directory(
            SLICE_EXTRACTION_DATASET_PATH,
            labels="inferred",
            label_mode="binary",
            image_size=IMAGE_SIZE,
            validation_split=0.2,
            subset="training",
            seed=1337
        ).prefetch(buffer_size=AUTOTUNE)

        validation_ds = image_dataset_from_directory(
            SLICE_EXTRACTION_DATASET_PATH,
            labels="inferred",
            label_mode="binary",
            image_size=IMAGE_SIZE,
            validation_split=0.2,
            subset="validation",
            seed=1337
        ).prefetch(buffer_size=AUTOTUNE)

        with STRATEGY.scope():
            base_model = InceptionV3(input_shape=(
                *IMAGE_SIZE, 3), include_top=False, weights='imagenet')

            for layer in base_model.layers:
                layer.trainable = False

            x = tf.keras.layers.Flatten()(base_model.output)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)

            # Add a final sigmoid layer with 1 node for classification output
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

            model = tf.keras.models.Model(base_model.input, x)

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Defining Callbacks
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{base_dir}/model.h5", monitor='val_loss', save_best_only=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            patience=10, monitor='val_accuracy', factor=0.6, min_lr=0.0000001)

        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=EPOCHS,
            callbacks=[save_best, reduce_lr]
        )
