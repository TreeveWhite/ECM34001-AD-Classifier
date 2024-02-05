import tensorflow as tf

from config import IMAGE_SIZE
#------------------------------------DENSENET-201-------------------------------#

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

def dense_net():
    model = tf.keras.Sequential([
            tf.keras.Input(shape=(*IMAGE_SIZE, 1)),

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
                4, activation='softmax')
        ])
    return model

#------------------------------------VGGNET-------------------------------#
