import tensorflow as tf
from models.dqn_model import DQN
from models.mamba_model import MambaModel


class AnyTaskLearner(tf.keras.Model):
    def __init__(self, state_dim, action_dim, sequence_dim, hidden_units):
        super(AnyTaskLearner, self).__init__()

        # Initialize the DQN model for state-based processing
        self.dqn = DQN(state_dim, action_dim)

        # Initialize the Mamba model for sequence-based processing
        self.mamba = MambaModel(sequence_dim, hidden_units, action_dim)

        # Final layer to combine DQN and Mamba outputs
        self.final_dense = tf.keras.layers.Dense(action_dim, activation='linear')

    def call(self, state, sequence):
        # Get the output from DQN
        dqn_output = self.dqn(state)

        # Get the output from MambaModel
        mamba_output = self.mamba(sequence)

        # Combine outputs from both models
        combined = tf.keras.layers.Concatenate()([dqn_output, mamba_output])

        # Final decision layer
        return self.final_dense(combined)
