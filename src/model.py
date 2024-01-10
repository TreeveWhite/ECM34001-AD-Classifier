import datetime
import tensorflow as tf
import keras

STRATEGY = tf.distribute.get_strategy()
EPOCHS = 10
MODELS_BASE_PATH = "/home/white/uni_workspace/ecm3401-dissertation/ECM34001-AD-Classifier/models/"
MODEL_SAVE_PATH = MODELS_BASE_PATH+f"AD_Model_{datetime.datetime.now()}.h5"


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn


class CNN:
    def __init__(self, image_size, num_classes, load_model_path=None, model_save_path=MODEL_SAVE_PATH) -> None:
        self.image_size = image_size
        self.num_classes = num_classes
        self.model_save_path = model_save_path

        if not load_model_path:
            self.build()
        else:
            self.model = keras.models.load_model(load_model_path)

    def build(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def compile(self, metrics=[keras.metrics.AUC(name='auc')]):
        with STRATEGY.scope():
            self.model.compile(
                optimizer='adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=metrics
            )

    def train(self, train_ds, val_ds):
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            exponential_decay(0.01, 20))
        checkpoint_cb = keras.callbacks.ModelCheckpoint(self.model_save_path,
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True)
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
            epochs=EPOCHS
        )

    def test(self, test_ds):
        self.model.evaluate(test_ds)

    def __repr__(self) -> str:
        return self.model.summary()
