
# from keras.models import load_model
import tensorflow as tf
# import tensorflow.keras

def load_model_from_h5(file_path):
    print("load")
    return tf.keras.models.load_model(file_path)
