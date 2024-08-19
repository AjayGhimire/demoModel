import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseModel

class DQN(BaseModel):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.q_values = layers.Dense(action_dim, activation='linear')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q_values = self.q_values(x)
        return q_values
