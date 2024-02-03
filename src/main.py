import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
import keras
import tensorflow_addons as tfa

from config import IMAGE_SIZE, CLASS_NAMES

STRATEGY = tf.distribute.get_strategy()
AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_SLICE"
BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync
EPOCHS = 10


def convolutional_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(
            filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )


def dense_block(units, dropout_rate):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])


if __name__ == "__main__":

    print("REPLICAS: ", STRATEGY.num_replicas_in_sync)

    train_ds = image_dataset_from_directory(
        DATASET_PATH,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        # color_mode="grayscale",
        validation_split=0.2,
        subset="training",
        seed=1337
    ).prefetch(buffer_size=AUTOTUNE)

    validation_ds = image_dataset_from_directory(
        DATASET_PATH,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        # color_mode="grayscale",
        validation_split=0.2,
        subset="validation",
        seed=1337
    ).prefetch(buffer_size=AUTOTUNE)

    print(f"Recognised Classes: {train_ds.class_names}")

    with STRATEGY.scope():
        # base_model = tf.keras.applications.DenseNet201(
        #     include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))

        # for layer in base_model.layers:
        #     layer.trainable = False

        # model = keras.models.Sequential([
        #     base_model,
        #     keras.layers.GlobalAveragePooling2D(),
        #     tfa.layers.WeightNormalization(
        #         keras.layers.Dense(256, activation='relu')),
        #     keras.layers.Dropout(0.5),
        #     tfa.layers.WeightNormalization(keras.layers.Dense(
        #         len(train_ds.class_names), activation='softmax'))
        # ])

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),

            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(),

            convolutional_block(32),
            convolutional_block(64),

            convolutional_block(128),
            tf.keras.layers.Dropout(0.2),

            convolutional_block(256),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            dense_block(512, 0.7),
            dense_block(128, 0.5),
            dense_block(64, 0.3),

            tf.keras.layers.Dense(
                len(train_ds.class_names), activation='softmax')
        ])

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Defining Callbacks
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=10, monitor='val_accuracy', factor=0.6, min_lr=0.0000001)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback, save_best,
                   reduce_lr]
    )
