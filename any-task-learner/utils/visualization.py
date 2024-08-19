import logging
import os


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('AnyTaskLearner')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger
