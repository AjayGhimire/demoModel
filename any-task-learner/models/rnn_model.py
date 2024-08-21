import tensorflow as tf


class SelectiveSSMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units):
        super(SelectiveSSMLayer, self).__init__()
        self.hidden_units = hidden_units

        # RNN with gates (like LSTM but with selectivity)
        self.state_update = tf.keras.layers.GRU(hidden_units, return_sequences=True)

        # Gating mechanisms
        self.selection_gate = tf.keras.layers.Dense(hidden_units, activation='sigmoid')

    def call(self, inputs):
        # Step 1: Pass inputs through the RNN layer
        rnn_output = self.state_update(inputs)

        # Step 2: Apply a gating mechanism to select relevant information
        gate_values = self.selection_gate(inputs)

        # Step 3: Element-wise multiplication of gate values and RNN output to filter out irrelevant information
        gated_output = gate_values * rnn_output

        return gated_output


class MambaModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MambaModel, self).__init__()

        # Selective State Space Model (SSM) layer for sequence processing
        self.ssm_layer = SelectiveSSMLayer(hidden_units)

        # Convolutional layer for additional sequence processing (local context modeling)
        self.conv_layer = tf.keras.layers.Conv1D(filters=hidden_units, kernel_size=3, padding='same')

        # Gating layer to control information flow after convolution
        self.conv_gate = tf.keras.layers.Dense(hidden_units, activation='sigmoid')

        # Final output layer for predictions
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, sequence):
        # Pass through the SSM layer
        x = self.ssm_layer(sequence)

        # Pass through the convolutional layer
        conv_output = self.conv_layer(x)

        # Apply a gating mechanism to selectively filter the convolutional output
        gate_values = self.conv_gate(x)
        gated_conv_output = gate_values * conv_output

        # Generate final output
        output = self.output_layer(gated_conv_output)

        return output
