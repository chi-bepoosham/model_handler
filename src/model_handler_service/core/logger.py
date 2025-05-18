import logging
import os
from datetime import datetime
from model_handler_service.core.config import config

# Get the directory path for logs from the configuration
log_dir = config.logs_path

# Configure the root logger
logging.basicConfig(
    level=logging.NOTSET,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Setup for a specific logger named 'model_handler' ---

model_log_file = os.path.join(log_dir, f'model_loading_{datetime.now().strftime("%Y%m%d")}.log')

# Create a file handler and set its level from config
file_handler = logging.FileHandler(model_log_file)
file_handler.setLevel(config.file_log_level) 
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create a logger for model loading
model_logger = logging.getLogger('model_handler')
model_logger.setLevel(logging.NOTSET)
model_logger.propagate = False 
model_logger.addHandler(file_handler)

# Create a console handler and set its level from config
console_handler = logging.StreamHandler()
console_handler.setLevel(config.console_log_level) # Use level from config
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
model_logger.addHandler(console_handler)
