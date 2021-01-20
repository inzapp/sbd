import tensorflow as tf

from model import Model


class Yolo:
    def __init__(self):
        self.model = tf.keras.models.Model()

    def fit(self, train_image_path, input_shape, batch_size, lr, epochs, validation_split=0.0, validation_image_path=None):
        self.model = Model(input_shape, output_channel).build()
        pass

    def load_model(self, model_path):
        pass

    def predict(self, img):
        pass
