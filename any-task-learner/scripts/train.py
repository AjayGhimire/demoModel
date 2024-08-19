import tensorflow as tf
from src.any_task_learner import AnyTaskLearner

# Configuration
state_dim = 10
action_dim = 4
sequence_dim = 5
hidden_units = 128
epochs = 10

# Load your data here (for now, we use random data)
state_data = tf.random.normal((100, state_dim))
sequence_data = tf.random.normal((100, sequence_dim))
actions = tf.random.uniform((100,), maxval=action_dim, dtype=tf.int32)
rewards = tf.random.uniform((100,))
next_state_data = tf.random.normal((100, state_dim))
next_sequence_data = tf.random.normal((100, sequence_dim))
done = tf.random.uniform((100,), maxval=2, dtype=tf.int32)

# Initialize the model
model = AnyTaskLearner(state_dim, action_dim, sequence_dim, hidden_units)
model.compile(optimizer='adam', loss='mse')

# Training loop
for epoch in range(epochs):
    model.train_step((state_data, sequence_data, actions, rewards, next_state_data, next_sequence_data, done))
    print(f'Epoch {epoch+1}/{epochs} complete')
