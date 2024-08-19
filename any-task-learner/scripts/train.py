import tensorflow as tf
import yaml
from models.any_task_learner import AnyTaskLearner
from utils.data_processing import load_data
from utils.logger import setup_logger

# Load configuration
with open('config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Setup logger
logger = setup_logger(config['paths']['log_dir'])

# Load data
state_data, sequence_data, actions, rewards, next_state_data, next_sequence_data, done = load_data(config['paths']['data_dir'])

# Initialize the model
model = AnyTaskLearner(
    state_dim=config['model']['state_dim'],
    action_dim=config['model']['action_dim'],
    sequence_dim=config['model']['sequence_dim'],
    hidden_units=config['model']['hidden_units']
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate']), loss='mse')

# Training loop
for epoch in range(config['model']['epochs']):
    model.train_step((state_data, sequence_data, actions, rewards, next_state_data, next_sequence_data, done))
    logger.info(f'Epoch {epoch+1}/{config["model"]["epochs"]} complete')

# Save the trained model
model.save_model(config['paths']['model_dir'] + '/any_task_learner.h5')
