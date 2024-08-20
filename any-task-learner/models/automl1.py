import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class AutoML1_Unsupervised(tf.keras.Model):
    def __init__(self, input_dim, hidden_units):
        super(AutoML1_Unsupervised, self).__init__()

        # Preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=hidden_units)

        # Autoencoder for dimensionality reduction
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units // 2, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def preprocess(self, X):
        # Standardize and reduce dimensionality
        X_scaled = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled)
        return X_reduced

    def call(self, inputs):
        # Preprocess inputs
        inputs_preprocessed = self.preprocess(inputs)

        # Encode and decode using the autoencoder
        encoded = self.encoder(inputs_preprocessed)
        decoded = self.decoder(encoded)

        return decoded, encoded
