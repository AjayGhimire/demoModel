import unittest
import tensorflow as tf
from src.dqn_model import DQN

class TestDQNModel(unittest.TestCase):
    def test_dqn_output_shape(self):
        model = DQN(state_dim=10, action_dim=4)
        state = tf.random.normal((1, 10))
        q_values = model(state)
        self.assertEqual(q_values.shape, (1, 4))

if __name__ == "__main__":
    unittest.main()
