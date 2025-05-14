import logging
import os
import time
from functools import wraps
from datetime import datetime
from model_handler_service.core.config import config

# Create logs directory if it doesn't exist
log_dir = config.logs_path

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a file handler for the model loading logger
model_log_file = os.path.join(log_dir, f'model_loading_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(model_log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create a logger for model loading
model_logger = logging.getLogger('model_handler')
model_logger.setLevel(logging.INFO)
model_logger.addHandler(file_handler)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
model_logger.addHandler(console_handler)
