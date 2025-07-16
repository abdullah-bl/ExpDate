# config.py
"""
Configuration file for ExpDate project.
Centralizes all important parameters and settings for easy management.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    
    # Data paths
    train_data_path: str = "data/train"
    eval_data_path: str = "data/evaluation"
    annotations_file: str = "annotations.json"
    images_dir: str = "images"
    
    # Image preprocessing
    image_size: Tuple[int, int] = (800, 1333)
    normalize_mean: List[float] = (0.485, 0.456, 0.406)
    normalize_std: List[float] = (0.229, 0.224, 0.225)
    
    # Data augmentation (for training)
    random_horizontal_flip_p: float = 0.5
    random_rotation_degrees: int = 10
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    
    # DataLoader settings
    batch_size: int = 8
    num_workers: int = 0  # Set to 0 for MPS compatibility
    shuffle: bool = True
    pin_memory: bool = False  # Set to False for MPS
    
    # Dummy data settings (for testing)
    dummy_num_samples: int = 100

@dataclass
class ModelConfig:
    """Model architecture and checkpoint configuration"""
    
    # Model checkpoint paths
    date_detector_checkpoint: str = "date_detector_mps.pth"
    dmy_detector_checkpoint: str = "dmy_detector_mps.pth"
    dan_recognizer_checkpoint: str = "dan_recognizer_mps.pth"
    
    # Date Detector settings
    date_detector_backbone: str = "resnet50"
    date_detector_fpn_channels: int = 256
    date_detector_num_classes: int = 4  # date, due, prod, code
    
    # DMY Detector settings
    dmy_detector_backbone: str = "resnet18"
    dmy_detector_fpn_channels: int = 128
    dmy_detector_num_classes: int = 3  # day, month, year
    
    # DAN Recognizer settings
    dan_recognizer_backbone: str = "resnet18"
    dan_recognizer_hidden_size: int = 256
    dan_recognizer_num_layers: int = 2
    dan_recognizer_dropout: float = 0.1
    
    # Vocabulary for text recognition
    vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    vocab_size: int = 38  # len(vocab) + 1 for padding
    max_text_length: int = 20

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    
    # General training settings
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9
    
    # Learning rate scheduling
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    
    # Loss function weights
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    ctr_loss_weight: float = 1.0
    
    # Checkpoint settings
    save_every_n_epochs: int = 5
    save_best_model: bool = True
    early_stopping_patience: int = 10
    
    # Validation settings
    validation_frequency: int = 1
    validation_split: float = 0.2

@dataclass
class DeviceConfig:
    """Device and hardware configuration"""
    
    # Device selection
    device: str = "auto"  # "auto", "mps", "cuda", "cpu"
    dtype: str = "float32"  # "float32", "float16"
    
    # Memory management
    clear_memory_frequency: int = 1  # Clear memory every N epochs
    max_memory_usage: float = 8.0  # GB
    
    # Mixed precision training
    use_mixed_precision: bool = True
    amp_dtype: str = "float16"

@dataclass
class InferenceConfig:
    """Inference and evaluation configuration"""
    
    # Confidence thresholds
    date_detection_threshold: float = 0.5
    dmy_detection_threshold: float = 0.5
    text_recognition_threshold: float = 0.3
    
    # NMS settings
    nms_threshold: float = 0.5
    max_detections: int = 100
    
    # Post-processing
    min_box_size: int = 10
    max_box_size: int = 1000
    
    # Output settings
    output_format: str = "DD/MM/YYYY"
    return_confidence: bool = True
    return_bbox: bool = False

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "expdate_training.log"
    tensorboard_log_dir: str = "logs"
    
    # Progress reporting
    print_every_n_batches: int = 10
    save_training_plots: bool = True
    plot_save_dir: str = "plots"
    
    # Metrics tracking
    track_metrics: List[str] = None
    
    def __post_init__(self):
        if self.track_metrics is None:
            self.track_metrics = ["loss", "accuracy", "precision", "recall"]

@dataclass
class PathConfig:
    """Path and directory configuration"""
    
    # Base directories
    project_root: str = "."
    models_dir: str = "models"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    
    # Create directories if they don't exist
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.models_dir,
            self.checkpoints_dir,
            self.results_dir,
            os.path.join(self.results_dir, "plots"),
            os.path.join(self.results_dir, "logs")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

class ExpDateConfig:
    """Main configuration class that combines all sub-configs"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to a JSON config file
        """
        # Initialize all sub-configs
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.device = DeviceConfig()
        self.inference = InferenceConfig()
        self.logging = LoggingConfig()
        self.paths = PathConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Create necessary directories
        self.paths.create_directories()
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        import json
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update sub-configs
            for section, values in config_dict.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str):
        """Save current configuration to JSON file"""
        import json
        try:
            config_dict = {}
            for section_name in ['data', 'model', 'training', 'device', 'inference', 'logging', 'paths']:
                section = getattr(self, section_name)
                config_dict[section_name] = {
                    key: value for key, value in section.__dict__.items()
                    if not key.startswith('_')
                }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def get_device_info(self):
        """Get current device information"""
        import torch
        
        if self.device.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.device.device)
    
    def get_dtype_info(self):
        """Get current dtype information"""
        import torch
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64
        }
        
        return dtype_map.get(self.device.dtype, torch.float32)
    
    def print_summary(self):
        """Print a summary of the current configuration"""
        print("=" * 50)
        print("ExpDate Configuration Summary")
        print("=" * 50)
        
        print(f"\nDevice: {self.get_device_info()}")
        print(f"Data Type: {self.get_dtype_info()}")
        
        print(f"\nData:")
        print(f"  Train Path: {self.data.train_data_path}")
        print(f"  Eval Path: {self.data.eval_data_path}")
        print(f"  Batch Size: {self.data.batch_size}")
        print(f"  Image Size: {self.data.image_size}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Weight Decay: {self.training.weight_decay}")
        
        print(f"\nModels:")
        print(f"  Date Detector: {self.model.date_detector_checkpoint}")
        print(f"  DMY Detector: {self.model.dmy_detector_checkpoint}")
        print(f"  DAN Recognizer: {self.model.dan_recognizer_checkpoint}")
        
        print(f"\nInference:")
        print(f"  Date Detection Threshold: {self.inference.date_detection_threshold}")
        print(f"  DMY Detection Threshold: {self.inference.dmy_detection_threshold}")
        print(f"  Text Recognition Threshold: {self.inference.text_recognition_threshold}")
        
        print("=" * 50)

# Default configuration instance
config = ExpDateConfig()

# Convenience functions for backward compatibility
def get_config() -> ExpDateConfig:
    """Get the default configuration instance"""
    return config

def update_config(**kwargs):
    """Update configuration with new values"""
    for section, values in kwargs.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)

# Example usage:
if __name__ == "__main__":
    # Print current configuration
    config.print_summary()
    
    # Example: Update some settings
    update_config(
        training={"num_epochs": 20, "learning_rate": 1e-3},
        data={"batch_size": 16},
        device={"device": "mps"}
    )
    
    print("\nAfter updates:")
    config.print_summary()
    
    # Save configuration
    config.save_to_file("config.json") 