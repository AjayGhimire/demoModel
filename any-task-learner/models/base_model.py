import tensorflow as tf


class BaseModel(tf.keras.Model):
    def save_model(self, file_path):
        self.save_weights(file_path)

    def load_model(self, file_path):
        self.load_weights(file_path)
