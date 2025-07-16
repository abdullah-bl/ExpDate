# mps_utils.py
import torch
import warnings
import os
from typing import Union, Optional

# Device detection with fallback
def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Global device and dtype settings
DEVICE = get_device()
# Use float32 for MPS to avoid compatibility issues
DTYPE = torch.float32

print(f"Running on {DEVICE} with dtype {DTYPE}")

def to_device(x: Union[torch.Tensor, dict, list, tuple], 
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move tensors to device with optional dtype conversion.
    Handles nested structures (dict, list, tuple).
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE
    
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype, non_blocking=True)
    elif isinstance(x, dict):
        return {k: to_device(v, device, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device, dtype) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_device(v, device, dtype) for v in x)
    else:
        return x

def model_to_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Move model to device with proper dtype handling for MPS"""
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    
    # For MPS, ensure model parameters are in the correct dtype
    if device.type == "mps":
        model = model.to(dtype=DTYPE)
    
    return model

def create_optimizer(model: torch.nn.Module, lr: float = 1e-3, 
                    momentum: float = 0.9, weight_decay: float = 1e-4):
    """Create optimizer with device-aware parameters"""
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    return optimizer

def create_scheduler(optimizer, step_size: int = 7, gamma: float = 0.1):
    """Create learning rate scheduler"""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Mixed precision training utilities
class MixedPrecisionTrainer:
    """Helper class for mixed precision training on MPS"""
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.model = model_to_device(model, self.device)
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
    
    def train_step(self, data, target, optimizer, loss_fn):
        """Single training step with mixed precision if available"""
        optimizer.zero_grad()
        
        if self.scaler is not None:
            # CUDA mixed precision
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = loss_fn(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # MPS or CPU training
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        return loss.item()

# Memory management utilities
def clear_memory():
    """Clear GPU/MPS memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
        }
    elif torch.backends.mps.is_available():
        return {
            'allocated': torch.mps.current_allocated_memory() / 1024**3,  # GB
            'cached': torch.mps.driver_allocated_memory() / 1024**3,      # GB
        }
    else:
        return {'allocated': 0, 'cached': 0}