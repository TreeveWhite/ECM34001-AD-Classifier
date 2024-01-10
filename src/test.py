import tensorflow as tf


MODELS_PATH = "/home/white/uni_workspace/ecm3401-dissertation/ECM34001-AD-Classifier/models/AD_Model_2024-01-10 11:37:44.993427.h5"


def preprocess_input(image_path, target_size=[176, 208]):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    return img_array


def nicify_prediction(prediction):
    return [round(prob * 100, 2) for prob in prediction[0]]


if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model(MODELS_PATH)
    loaded_model.summary()

    image_path = "/home/white/uni_workspace/ecm3401-dissertation/data/example_ad_dataset/test/ModerateDemented/27.jpg"

    preprocessed_image = preprocess_input(image_path)
