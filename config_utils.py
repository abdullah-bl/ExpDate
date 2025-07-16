# config_utils.py
"""
Utility functions for working with the ExpDate configuration system.
Provides helper functions for common configuration tasks and examples.
"""

import os
import json
from typing import Dict, Any, Optional
from config import ExpDateConfig, get_config, update_config

def create_training_config(
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    device: str = "auto"
) -> ExpDateConfig:
    """
    Create a configuration optimized for training.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        device: Device to use for training
        
    Returns:
        Configured ExpDateConfig instance
    """
    config = ExpDateConfig()
    
    # Update training parameters
    config.training.num_epochs = num_epochs
    config.training.learning_rate = learning_rate
    config.data.batch_size = batch_size
    config.device.device = device
    
    # Enable logging and checkpointing
    config.logging.save_training_plots = True
    config.training.save_best_model = True
    config.training.save_every_n_epochs = 5
    
    return config

def create_inference_config(
    confidence_threshold: float = 0.5,
    device: str = "auto",
    return_confidence: bool = True
) -> ExpDateConfig:
    """
    Create a configuration optimized for inference.
    
    Args:
        confidence_threshold: Confidence threshold for detections
        device: Device to use for inference
        return_confidence: Whether to return confidence scores
        
    Returns:
        Configured ExpDateConfig instance
    """
    config = ExpDateConfig()
    
    # Update inference parameters
    config.inference.date_detection_threshold = confidence_threshold
    config.inference.dmy_detection_threshold = confidence_threshold
    config.inference.text_recognition_threshold = confidence_threshold
    config.device.device = device
    config.inference.return_confidence = return_confidence
    
    # Disable training-specific features
    config.logging.save_training_plots = False
    config.training.save_best_model = False
    
    return config

def create_fast_training_config() -> ExpDateConfig:
    """
    Create a configuration for fast training (smaller models, fewer epochs).
    
    Returns:
        Configured ExpDateConfig instance
    """
    config = ExpDateConfig()
    
    # Reduce model complexity
    config.model.date_detector_backbone = "resnet18"
    config.model.dmy_detector_backbone = "resnet18"
    config.model.dan_recognizer_backbone = "resnet18"
    
    # Reduce training time
    config.training.num_epochs = 5
    config.data.batch_size = 16
    config.training.learning_rate = 1e-3
    
    # Reduce image size for faster processing
    config.data.image_size = (400, 667)
    
    return config

def create_high_accuracy_config() -> ExpDateConfig:
    """
    Create a configuration for high accuracy (larger models, more training).
    
    Returns:
        Configured ExpDateConfig instance
    """
    config = ExpDateConfig()
    
    # Use larger models
    config.model.date_detector_backbone = "resnet50"
    config.model.dmy_detector_backbone = "resnet34"
    config.model.dan_recognizer_backbone = "resnet34"
    
    # Increase training time
    config.training.num_epochs = 30
    config.training.learning_rate = 5e-5
    config.data.batch_size = 4
    
    # Use larger image size
    config.data.image_size = (1024, 1707)
    
    # More aggressive data augmentation
    config.data.random_rotation_degrees = 15
    config.data.color_jitter_brightness = 0.3
    config.data.color_jitter_contrast = 0.3
    
    return config

def load_config_from_file(config_path: str) -> ExpDateConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Loaded ExpDateConfig instance
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return ExpDateConfig(config_path)

def save_config_to_file(config: ExpDateConfig, config_path: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: ExpDateConfig instance to save
        config_path: Path where to save the configuration file
    """
    config.save_to_file(config_path)

def merge_configs(base_config: ExpDateConfig, override_config: Dict[str, Any]) -> ExpDateConfig:
    """
    Merge a base configuration with override values.
    
    Args:
        base_config: Base configuration to start with
        override_config: Dictionary of override values
        
    Returns:
        Merged ExpDateConfig instance
    """
    # Create a copy of the base config
    merged_config = ExpDateConfig()
    
    # Copy all values from base config
    for section_name in ['data', 'model', 'training', 'device', 'inference', 'logging', 'paths']:
        base_section = getattr(base_config, section_name)
        merged_section = getattr(merged_config, section_name)
        
        for key, value in base_section.__dict__.items():
            if not key.startswith('_'):
                setattr(merged_section, key, value)
    
    # Apply overrides
    for section, values in override_config.items():
        if hasattr(merged_config, section):
            section_config = getattr(merged_config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    return merged_config

def validate_config(config: ExpDateConfig) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: ExpDateConfig instance to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Check data paths
    if not os.path.exists(config.data.train_data_path):
        errors.append(f"Training data path does not exist: {config.data.train_data_path}")
    
    if not os.path.exists(config.data.eval_data_path):
        errors.append(f"Evaluation data path does not exist: {config.data.eval_data_path}")
    
    # Check model parameters
    if config.model.vocab_size != len(config.model.vocab) + 1:
        errors.append("Vocabulary size should be len(vocab) + 1 (for padding)")
    
    # Check training parameters
    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.training.num_epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.data.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Check device settings
    valid_devices = ["auto", "mps", "cuda", "cpu"]
    if config.device.device not in valid_devices:
        errors.append(f"Device must be one of: {valid_devices}")
    
    # Check inference thresholds
    if not (0 <= config.inference.date_detection_threshold <= 1):
        errors.append("Date detection threshold must be between 0 and 1")
    
    if not (0 <= config.inference.dmy_detection_threshold <= 1):
        errors.append("DMY detection threshold must be between 0 and 1")
    
    if not (0 <= config.inference.text_recognition_threshold <= 1):
        errors.append("Text recognition threshold must be between 0 and 1")
    
    # Report errors
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration validation passed!")
    return True

def print_config_diff(config1: ExpDateConfig, config2: ExpDateConfig):
    """
    Print differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
    """
    print("Configuration differences:")
    print("=" * 50)
    
    for section_name in ['data', 'model', 'training', 'device', 'inference', 'logging', 'paths']:
        section1 = getattr(config1, section_name)
        section2 = getattr(config2, section_name)
        
        differences = []
        for key, value1 in section1.__dict__.items():
            if not key.startswith('_'):
                value2 = getattr(section2, key)
                if value1 != value2:
                    differences.append(f"  {key}: {value1} -> {value2}")
        
        if differences:
            print(f"\n{section_name.upper()}:")
            for diff in differences:
                print(diff)

def get_config_summary(config: ExpDateConfig) -> Dict[str, Any]:
    """
    Get a summary of the configuration as a dictionary.
    
    Args:
        config: ExpDateConfig instance
        
    Returns:
        Dictionary containing configuration summary
    """
    summary = {
        "device": str(config.get_device_info()),
        "dtype": str(config.get_dtype_info()),
        "data": {
            "train_path": config.data.train_data_path,
            "eval_path": config.data.eval_data_path,
            "batch_size": config.data.batch_size,
            "image_size": config.data.image_size
        },
        "training": {
            "epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay
        },
        "models": {
            "date_detector": config.model.date_detector_checkpoint,
            "dmy_detector": config.model.dmy_detector_checkpoint,
            "dan_recognizer": config.model.dan_recognizer_checkpoint
        },
        "inference": {
            "date_threshold": config.inference.date_detection_threshold,
            "dmy_threshold": config.inference.dmy_detection_threshold,
            "text_threshold": config.inference.text_recognition_threshold
        }
    }
    
    return summary

# Example usage functions
def example_training_setup():
    """Example of setting up configuration for training"""
    print("Setting up configuration for training...")
    
    # Create training configuration
    config = create_training_config(
        num_epochs=20,
        learning_rate=1e-4,
        batch_size=8,
        device="mps"
    )
    
    # Validate configuration
    if validate_config(config):
        # Save configuration
        config.save_to_file("training_config.json")
        print("Training configuration saved to training_config.json")
        
        # Print summary
        config.print_summary()
    
    return config

def example_inference_setup():
    """Example of setting up configuration for inference"""
    print("Setting up configuration for inference...")
    
    # Create inference configuration
    config = create_inference_config(
        confidence_threshold=0.7,
        device="mps",
        return_confidence=True
    )
    
    # Validate configuration
    if validate_config(config):
        # Save configuration
        config.save_to_file("inference_config.json")
        print("Inference configuration saved to inference_config.json")
        
        # Print summary
        config.print_summary()
    
    return config

if __name__ == "__main__":
    # Run examples
    print("ExpDate Configuration Utilities")
    print("=" * 50)
    
    # Example 1: Training setup
    print("\n1. Training Configuration Example:")
    training_config = example_training_setup()
    
    # Example 2: Inference setup
    print("\n2. Inference Configuration Example:")
    inference_config = example_inference_setup()
    
    # Example 3: Fast training setup
    print("\n3. Fast Training Configuration Example:")
    fast_config = create_fast_training_config()
    fast_config.print_summary()
    
    # Example 4: High accuracy setup
    print("\n4. High Accuracy Configuration Example:")
    high_acc_config = create_high_accuracy_config()
    high_acc_config.print_summary()
    
    # Example 5: Configuration comparison
    print("\n5. Configuration Comparison:")
    print_config_diff(training_config, inference_config) 