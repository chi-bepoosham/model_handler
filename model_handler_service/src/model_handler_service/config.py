import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

class Config:
    """Configuration handler for the model handler service."""
    
    # Required environment variables
    REQUIRED_VARS = [
        'TEMP_IMAGES_DIR'
    ]

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Load environment variables from .env file if it exists
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
        
        # Validate required environment variables
        self._validate_env_vars()
        
        # Initialize configuration
        self._init_config()

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_VARS if not os.getenv(var)]
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _init_config(self) -> None:
        """Initialize configuration attributes."""
        # Image Processing Configuration
        base_path = os.path.dirname(__file__)
        self.temp_images_dir = Path(os.path.join(base_path, '../../temp_images'))
        
        # Service Configuration
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        # Model Configuration
        self.model_path = Path(os.getenv('MODEL_PATH', 'models'))
        self.male_model_version = os.getenv('MALE_MODEL_VERSION', '1.0')
        self.female_model_version = os.getenv('FEMALE_MODEL_VERSION', '1.0')

        # RabbitMQ Configuration
        self._rabbitmq_config = {
            'host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'user': os.getenv('RABBITMQ_USER', 'guest'),
            'password': os.getenv('RABBITMQ_PASSWORD', 'guest'),
            'vhost': os.getenv('RABBITMQ_VHOST', '/'),
            'response_queue': os.getenv('RABBITMQ_RESPONSE_QUEUE', 'model_response')
        }

        # Ensure temp images directory exists
        self.temp_images_dir.mkdir(parents=True, exist_ok=True)

    def get_temp_dir(self) -> Path:
        """Get temporary images directory path."""
        return self.temp_images_dir

    def get_rabbitmq_config(self) -> Dict[str, str]:
        """Get RabbitMQ configuration."""
        return self._rabbitmq_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'temp_images_dir': str(self.temp_images_dir),
            'debug': self.debug,
            'log_level': self.log_level,
            'model_path': str(self.model_path),
            'male_model_version': self.male_model_version,
            'female_model_version': self.female_model_version,
            'rabbitmq': self._rabbitmq_config
        }

# Create a global configuration instance
config = Config() 