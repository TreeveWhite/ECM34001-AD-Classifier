import sys
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from datetime import datetime
import os

from cnn_models import dense_net

from config import IMAGE_SIZE, MODEL_SAVE_PATH, DATASET_PATH

STRATEGY = tf.distribute.get_strategy()
AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync
EPOCHS = 10

MODEL_NAME = "DENSENET"


if __name__ == "__main__":
    base_dir = f"{MODEL_SAVE_PATH}/{MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    os.makedirs(base_dir, exist_ok=True)

    with open(f"{base_dir}/log.log", "w") as sys.stdout:

        print("REPLICAS: ", STRATEGY.num_replicas_in_sync)

        train_ds = image_dataset_from_directory(
            DATASET_PATH,
            labels="inferred",
            label_mode="categorical",
            image_size=IMAGE_SIZE,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=1337
        ).prefetch(buffer_size=AUTOTUNE)

        validation_ds = image_dataset_from_directory(
            DATASET_PATH,
            labels="inferred",
            label_mode="categorical",
            image_size=IMAGE_SIZE,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=1337
        ).prefetch(buffer_size=AUTOTUNE)

        with STRATEGY.scope():
            model = dense_net()

        model.summary()

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
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
        