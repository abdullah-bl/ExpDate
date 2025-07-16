# ExpDate Configuration System

This document explains how to use the centralized configuration system for the ExpDate project, which makes it easy to manage all parameters and settings in one place.

## Overview

The configuration system consists of three main files:
- `config.py` - Main configuration classes and default settings
- `config_utils.py` - Utility functions for working with configurations
- `config_example.json` - Example JSON configuration file

## Quick Start

### 1. Basic Usage

```python
from config import get_config

# Get the default configuration
config = get_config()

# Access configuration values
print(f"Batch size: {config.data.batch_size}")
print(f"Learning rate: {config.training.learning_rate}")
print(f"Device: {config.get_device_info()}")
```

### 2. Print Configuration Summary

```python
from config import get_config

config = get_config()
config.print_summary()
```

### 3. Update Configuration

```python
from config import update_config

# Update multiple settings at once
update_config(
    training={"num_epochs": 20, "learning_rate": 1e-3},
    data={"batch_size": 16},
    device={"device": "mps"}
)
```

## Configuration Sections

### Data Configuration (`config.data`)

Controls data loading and preprocessing:

```python
config.data.train_data_path = "data/train"
config.data.eval_data_path = "data/evaluation"
config.data.batch_size = 8
config.data.image_size = (800, 1333)
config.data.normalize_mean = [0.485, 0.456, 0.406]
config.data.normalize_std = [0.229, 0.224, 0.225]
```

**Key Parameters:**
- `train_data_path`: Path to training data
- `eval_data_path`: Path to evaluation data
- `batch_size`: Batch size for training/inference
- `image_size`: Input image dimensions (height, width)
- `normalize_mean/std`: Image normalization values
- `random_horizontal_flip_p`: Probability of horizontal flip during training
- `random_rotation_degrees`: Maximum rotation angle for augmentation

### Model Configuration (`config.model`)

Controls model architecture and checkpoint paths:

```python
config.model.date_detector_checkpoint = "models/date_detector_mps.pth"
config.model.dmy_detector_checkpoint = "models/dmy_detector_mps.pth"
config.model.dan_recognizer_checkpoint = "models/dan_recognizer_mps.pth"
config.model.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
config.model.max_text_length = 20
```

**Key Parameters:**
- `date_detector_checkpoint`: Path to date detector model
- `dmy_detector_checkpoint`: Path to DMY detector model
- `dan_recognizer_checkpoint`: Path to DAN recognizer model
- `vocab`: Character vocabulary for text recognition
- `max_text_length`: Maximum text length for recognition

### Training Configuration (`config.training`)

Controls training hyperparameters:

```python
config.training.num_epochs = 15
config.training.learning_rate = 1e-4
config.training.weight_decay = 1e-5
config.training.momentum = 0.9
config.training.save_every_n_epochs = 5
config.training.save_best_model = True
```

**Key Parameters:**
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `weight_decay`: Weight decay for regularization
- `momentum`: Momentum for SGD optimizer
- `scheduler_step_size`: Learning rate scheduler step size
- `scheduler_gamma`: Learning rate decay factor

### Device Configuration (`config.device`)

Controls hardware and device settings:

```python
config.device.device = "auto"  # "auto", "mps", "cuda", "cpu"
config.device.dtype = "float32"  # "float32", "float16"
config.device.use_mixed_precision = True
config.device.max_memory_usage = 8.0  # GB
```

**Key Parameters:**
- `device`: Device to use ("auto" automatically selects best available)
- `dtype`: Data type for tensors
- `use_mixed_precision`: Enable mixed precision training
- `max_memory_usage`: Maximum memory usage limit

### Inference Configuration (`config.inference`)

Controls inference and evaluation settings:

```python
config.inference.date_detection_threshold = 0.5
config.inference.dmy_detection_threshold = 0.5
config.inference.text_recognition_threshold = 0.3
config.inference.nms_threshold = 0.5
config.inference.output_format = "DD/MM/YYYY"
```

**Key Parameters:**
- `date_detection_threshold`: Confidence threshold for date detection
- `dmy_detection_threshold`: Confidence threshold for DMY detection
- `text_recognition_threshold`: Confidence threshold for text recognition
- `nms_threshold`: Non-maximum suppression threshold
- `output_format`: Output date format

### Logging Configuration (`config.logging`)

Controls logging and monitoring:

```python
config.logging.log_level = "INFO"
config.logging.log_file = "expdate_training.log"
config.logging.save_training_plots = True
config.logging.print_every_n_batches = 10
```

## Predefined Configurations

The `config_utils.py` file provides several predefined configurations for common use cases:

### 1. Training Configuration

```python
from config_utils import create_training_config

config = create_training_config(
    num_epochs=20,
    learning_rate=1e-4,
    batch_size=8,
    device="mps"
)
```

### 2. Inference Configuration

```python
from config_utils import create_inference_config

config = create_inference_config(
    confidence_threshold=0.7,
    device="mps",
    return_confidence=True
)
```

### 3. Fast Training Configuration

```python
from config_utils import create_fast_training_config

config = create_fast_training_config()  # Smaller models, fewer epochs
```

### 4. High Accuracy Configuration

```python
from config_utils import create_high_accuracy_config

config = create_high_accuracy_config()  # Larger models, more training
```

## Loading and Saving Configurations

### Load from JSON File

```python
from config import ExpDateConfig

# Load configuration from JSON file
config = ExpDateConfig("my_config.json")
```

### Save to JSON File

```python
from config import get_config

config = get_config()
config.save_to_file("my_config.json")
```

### Example JSON Configuration

```json
{
  "data": {
    "batch_size": 16,
    "image_size": [800, 1333]
  },
  "training": {
    "num_epochs": 20,
    "learning_rate": 0.0001
  },
  "device": {
    "device": "mps"
  }
}
```

## Validation

The configuration system includes validation to catch common errors:

```python
from config_utils import validate_config
from config import get_config

config = get_config()
if validate_config(config):
    print("Configuration is valid!")
else:
    print("Configuration has errors!")
```

## Integration with Existing Code

To integrate the configuration system with your existing training scripts:

### Before (Hard-coded values):
```python
# dan_train.py
def train_dan(model, dataloader, num_epochs=15, lr=1e-4, save_path="dan_recognizer_mps.pth"):
    # ... training code
```

### After (Using configuration):
```python
# dan_train.py
from config import get_config

def train_dan(model, dataloader, config=None):
    if config is None:
        config = get_config()
    
    num_epochs = config.training.num_epochs
    lr = config.training.learning_rate
    save_path = config.model.dan_recognizer_checkpoint
    
    # ... training code using config values
```

## Best Practices

1. **Use Configuration Files**: Save different configurations for different scenarios (training, inference, testing)

2. **Validate Configurations**: Always validate configurations before using them

3. **Document Changes**: When modifying configurations, document the changes and their rationale

4. **Use Predefined Configurations**: Use the predefined configurations in `config_utils.py` for common scenarios

5. **Environment-Specific Configs**: Create different configuration files for different environments (development, production, testing)

## Example Workflows

### Training Workflow

```python
from config_utils import create_training_config, validate_config

# Create training configuration
config = create_training_config(
    num_epochs=20,
    learning_rate=1e-4,
    batch_size=8
)

# Validate configuration
if validate_config(config):
    # Save configuration
    config.save_to_file("training_config.json")
    
    # Use configuration in training
    train_model(config)
```

### Inference Workflow

```python
from config_utils import create_inference_config

# Create inference configuration
config = create_inference_config(
    confidence_threshold=0.7,
    return_confidence=True
)

# Use configuration in inference
results = run_inference(config)
```

### Configuration Comparison

```python
from config_utils import print_config_diff
from config_utils import create_training_config, create_inference_config

training_config = create_training_config()
inference_config = create_inference_config()

# Compare configurations
print_config_diff(training_config, inference_config)
```

## Troubleshooting

### Common Issues

1. **Configuration not found**: Make sure the JSON file exists and has valid syntax
2. **Invalid device**: Use "auto", "mps", "cuda", or "cpu" for device setting
3. **Invalid thresholds**: Ensure confidence thresholds are between 0 and 1
4. **Path not found**: Verify that data paths exist in your file system

### Debug Configuration

```python
from config import get_config

config = get_config()
config.print_summary()  # Print all current settings
```

This configuration system provides a centralized, flexible, and maintainable way to manage all parameters in your ExpDate project. It makes it easy to experiment with different settings, share configurations, and maintain consistency across different parts of your codebase. 