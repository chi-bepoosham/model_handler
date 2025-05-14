import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

def resolve_path(path_str: Optional[str], base_dir: Path = None) -> Path:
    """
    Resolve a path string to an absolute Path object, handling '..' and various formats.
    
    Args:
        path_str: The input path string (e.g., '/../data-models', 'temp_images', './relative', '~/home').
        base_dir: Base directory for relative paths (defaults to current working directory).
    
    Returns:
        A resolved absolute Path object.
    
    Raises:
        ConfigurationError: If the path is invalid or cannot be resolved.
    """
    if not path_str:
        raise ConfigurationError("Path cannot be empty or None")

    # Use current working directory if base_dir is not provided
    base_dir = base_dir or Path.cwd()

    # Expand user home directory (e.g., ~/path)
    path_str = os.path.expanduser(path_str)

    # Handle paths starting with '/..'
    if path_str.startswith('/../'):
        # Treat as relative to parent of current directory
        return (base_dir.parent / path_str[4:]).resolve()
    elif path_str.startswith('/'):
        # Absolute path
        return Path(path_str).resolve()

    # Handle relative paths (including those with '..' or without)
    try:
        return (base_dir / path_str).resolve()
    except (FileNotFoundError, RuntimeError) as e:
        raise ConfigurationError(f"Invalid path '{path_str}': {str(e)}")

class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

class Config:
    """Configuration handler for the model handler service."""
    
    # Required environment variables
    REQUIRED_VARS = [
        'TEMP_IMAGES_DIR',
        'MODEL_PATH',
        'LOGS_PATH' 
    ]

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Load environment variables from .env file if it exists
        # Try to find the .env file in the project root directory
        current_dir = Path(os.getcwd())
        env_path = current_dir / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        
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
        # Resolve temporary images directory
        temp_dir = os.getenv('TEMP_IMAGES_DIR')
        if not temp_dir:
            raise ConfigurationError("TEMP_IMAGES_DIR environment variable is not set")
        try:
            self.temp_images_dir = resolve_path(temp_dir)
        except ConfigurationError as e:
            raise ConfigurationError(f"Failed to resolve TEMP_IMAGES_DIR: {str(e)}")

        # Resolve model path
        model_path = os.getenv('MODEL_PATH')
        if not model_path:
            raise ConfigurationError("MODEL_PATH environment variable is not set")
        try:
            self.model_path = resolve_path(model_path)
        except ConfigurationError as e:
            raise ConfigurationError(f"Failed to resolve MODEL_PATH: {str(e)}")

        # Resolve logs path
        logs_path = os.getenv('LOGS_PATH')
        if not logs_path:
            raise ConfigurationError("LOGS_PATH environment variable is not set")
        try:
            self.logs_path = resolve_path(logs_path)
        except ConfigurationError as e:
            raise ConfigurationError(f"Failed to resolve LOGS_PATH: {str(e)}")

        # Ensure temp images directory exists and is writable
        try:
            self.temp_images_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno == 30:  # Read-only file system
                raise ConfigurationError(
                    f"Temporary images directory '{self.temp_images_dir}' is not writable. Please create it manually."
                )
            raise ConfigurationError(f"Failed to create directory '{self.temp_images_dir}': {str(e)}")
    
    def get_temp_dir(self) -> Path:
        """Get temporary images directory path."""
        return self.temp_images_dir

    def get_model_path(self) -> Path:
        """Get model path."""
        return self.model_path
        
    def get_model_file_path(self, model_subpath: str) -> Path:
        """Get the full path to a specific model file.
        
        Args:
            model_subpath: Relative path to the model file from the model directory
                          (e.g., 'astin/astinman.h5')
        
        Returns:
            Path: Full resolved path to the model file
        """
        return self.model_path / model_subpath

    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'temp_images_dir': str(self.temp_images_dir),
            'model_path': str(self.model_path),
            'logs_path': str(self.logs_path) 
        }

# Create a global configuration instance
config = Config() 