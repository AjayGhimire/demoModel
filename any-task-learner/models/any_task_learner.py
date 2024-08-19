import tensorflow as tf
from tensorflow.keras import layers
from .dqn_model import DQN
from .rnn_model import MambaLinearRNN

class AnyTaskLearner(tf.keras.Model):
    def __init__(self, state_dim, action_dim, sequence_dim, hidden_units):
        super(AnyTaskLearner, self).__init__()
        self.dqn = DQN(state_dim, action_dim)
        self.rnn = MambaLinearRNN(sequence_dim, hidden_units, hidden_units)
        self.final_dense = layers.Dense(action_dim, activation='linear')

    def call(self, state, sequence):
        dqn_output = self.dqn(state)
        rnn_output = self.rnn(sequence)
        combined = layers.Concatenate()([dqn_output, rnn_output])
        final_output = self.final_dense(combined)
        return final_output
