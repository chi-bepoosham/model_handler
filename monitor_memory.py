import psutil
import os
import time
import logging
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_memory_usage(stage):
    """Log current memory usage with stage name"""
    memory = get_memory_usage()
    logger.info(f"Memory usage at {stage}: {memory:.2f} MB")
    return memory

def monitor_imports():
    """Monitor memory usage during imports"""
    initial_memory = log_memory_usage("Initial (before any imports)")
    
    # Import and monitor core dependencies
    logger.info("\n=== Importing Core Dependencies ===")
    import tensorflow as tf
    core_memory = log_memory_usage("After TensorFlow import")
    logger.info(f"Memory increase from TensorFlow: {core_memory - initial_memory:.2f} MB")
    
    import torch
    torch_memory = log_memory_usage("After PyTorch import")
    logger.info(f"Memory increase from PyTorch: {torch_memory - core_memory:.2f} MB")
    
    # Clear any cached memory
    gc.collect()
    logger.info("Garbage collection performed")
    
    return log_memory_usage("After core imports and GC")

def monitor_model_loading():
    """Monitor memory usage during model loading"""
    logger.info("\n=== Starting Model Loading Monitor ===")
    base_memory = log_memory_usage("Before loading any models")
    
    # Import model loaders
    logger.info("\n=== Importing Model Handlers ===")
    from model_handler_service.load_and_predict_man import process_clothing_image, get_man_body_type
    man_import_memory = log_memory_usage("After importing male model handlers")
    logger.info(f"Memory increase from male handlers import: {man_import_memory - base_memory:.2f} MB")
    
    from model_handler_service.load_and_predict_woman import (
        process_woman_clothing_image, 
        process_six_model_predictions, 
        get_body_type_female
    )
    woman_import_memory = log_memory_usage("After importing female model handlers")
    logger.info(f"Memory increase from female handlers import: {woman_import_memory - man_import_memory:.2f} MB")
    
    # Monitor individual model loading
    logger.info("\n=== Loading Individual Models ===")
    
    # Test male models
    logger.info("\nTesting male models...")
    mem_before = log_memory_usage("Before male models test")
    test_img = "test_image.jpg"  # This is just for testing, won't actually process
    try:
        _ = process_clothing_image
        _ = get_man_body_type
    except Exception as e:
        logger.error(f"Error in male models: {str(e)}")
    mem_after = log_memory_usage("After male models test")
    logger.info(f"Memory increase from male models: {mem_after - mem_before:.2f} MB")
    
    # Test female models
    logger.info("\nTesting female models...")
    mem_before = log_memory_usage("Before female models test")
    try:
        _ = process_woman_clothing_image
        _ = get_body_type_female
        _ = process_six_model_predictions
    except Exception as e:
        logger.error(f"Error in female models: {str(e)}")
    mem_after = log_memory_usage("After female models test")
    logger.info(f"Memory increase from female models: {mem_after - mem_before:.2f} MB")
    
    # Final garbage collection
    gc.collect()
    final_memory = log_memory_usage("Final (after GC)")
    
    logger.info("\n=== Memory Usage Summary ===")
    logger.info(f"Initial memory usage: {base_memory:.2f} MB")
    logger.info(f"Final memory usage: {final_memory:.2f} MB")
    logger.info(f"Total memory increase: {final_memory - base_memory:.2f} MB")

if __name__ == "__main__":
    try:
        logger.info("Starting memory monitoring...")
        base_memory = monitor_imports()
        monitor_model_loading()
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")
        raise