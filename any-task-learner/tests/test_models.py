import unittest
import tensorflow as tf
from models.any_task_learner import AnyTaskLearner

class TestAnyTaskLearner(unittest.TestCase):
    def test_model_output_shape(self):
        model = AnyTaskLearner(state_dim=10, action_dim=4, sequence_dim=5, hidden_units=128)
        state = tf.random.normal((1, 10))
        sequence = tf.random.normal((1, 5))
        output = model(state, sequence)
        self.assertEqual(output.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()
