import tensorflow as tf

STRATEGY = tf.distribute.get_strategy()
EPOCHS = 1
MODELS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/ECM34001-AD-Classifier/models/"


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn


class CNN:
    def __init__(self, image_size, num_classes) -> None:
        self.image_size = image_size
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(*self.image_size, 3)),

            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(),

            self.__convolutional_block(32),
            self.__convolutional_block(64),

            self.__convolutional_block(128),
            tf.keras.layers.Dropout(0.2),

            self.__convolutional_block(256),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            self.__dense_block(512, 0.7),
            self.__dense_block(128, 0.5),
            self.__dense_block(64, 0.3),

            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def __convolutional_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                filters, 3, activation='relu', padding='same'),
            tf.keras.layers.SeparableConv2D(
                filters, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D()
        ])

    def __dense_block(self, units, dropout_rate):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    def compile(self, metrics=[tf.keras.metrics.AUC(name='auc')]):
        with STRATEGY.scope():
            self.model.compile(
                optimizer='adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=metrics
            )

    def train(self, train_ds, val_ds):
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            exponential_decay(0.01, 20))
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODELS_PATH+"alzheimer_model.h5",
                                                           save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                             restore_best_weights=True)
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
            epochs=EPOCHS
        )

    def test(self, test_ds):
        self.model.evaluate(test_ds)
