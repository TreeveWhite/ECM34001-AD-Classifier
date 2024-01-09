import tensorflow as tf

STRATEGY = tf.distribute.get_strategy()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync
IMAGE_SIZE = [176, 208]


class PreProcessor:
    def __init__(self, ds_training_path, ds_test_path, class_names) -> None:
        self.ds_training_path = ds_training_path
        self.ds_test_path = ds_test_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.load_dataset()

        self.encode_dataset()

    def load_dataset(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.ds_training_path,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.ds_training_path,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.ds_test_path,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )

        self.train_ds.class_names = self.class_names
        self.val_ds.class_names = self.class_names

    def one_hot_label(self, image, label):
        label = tf.one_hot(label, self.num_classes)
        return image, label

    def encode_dataset(self):
        self.train_ds = self.train_ds.map(
            self.one_hot_label, num_parallel_calls=AUTOTUNE)
        self.val_ds = self.val_ds.map(
            self.one_hot_label, num_parallel_calls=AUTOTUNE)
        self.test_ds = self.test_ds.map(
            self.one_hot_label, num_parallel_calls=AUTOTUNE)

        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
