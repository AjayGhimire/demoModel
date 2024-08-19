import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseModel

class MambaLinearRNN(BaseModel):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MambaLinearRNN, self).__init__()
        self.linear_transform = layers.Dense(hidden_units, activation='linear')
        self.rnn_layer = layers.SimpleRNN(hidden_units, activation='relu', return_sequences=False)
        self.output_layer = layers.Dense(output_dim, activation='relu')

    def call(self, sequence):
        x = self.linear_transform(sequence)
        x = self.rnn_layer(x)
        output = self.output_layer(x)
        return output
