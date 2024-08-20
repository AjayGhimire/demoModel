import tensorflow as tf
from tensorflow.keras import layers

class DQN(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_dim, activation='linear')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q_values = self.output_layer(x)
        return q_values
