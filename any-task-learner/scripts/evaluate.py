import tensorflow as tf
from src.any_task_learner import AnyTaskLearner

# Load the model
model = AnyTaskLearner(state_dim=10, action_dim=4, sequence_dim=5, hidden_units=128)
model.load_model("path_to_saved_model")

# Load evaluation data
evaluation_data = tf.random.normal((20, 10))

# Evaluate the model
predictions = model(evaluation_data)
print(predictions)
