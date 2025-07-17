# dmy_detector_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any

from train.dmy_detector import DMYDetector
from train.data import build_dataloader, create_dummy_dataloader
from utils.mps_utils import (
    model_to_device, to_device, create_optimizer, create_scheduler,
    clear_memory, get_memory_info, DEVICE, DTYPE
)

class DMYLoss(nn.Module):
    """Loss function for DMY detection"""
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.ctr_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, cls_logits, bbox_reg, centerness, targets):
        """Compute DMY detection loss"""
        # Extract target tensors from the new structure
        target_cls = to_device(targets['cls'])
        target_reg = to_device(targets['reg'])
        target_ctr = to_device(targets['ctr'])
        
        # Resize targets to match prediction size if needed
        if cls_logits.shape[-2:] != target_cls.shape[-2:]:
            target_cls = nn.functional.interpolate(
                target_cls, size=cls_logits.shape[-2:], mode='nearest'
            )
            target_reg = nn.functional.interpolate(
                target_reg, size=bbox_reg.shape[-2:], mode='nearest'
            )
            target_ctr = nn.functional.interpolate(
                target_ctr, size=centerness.shape[-2:], mode='nearest'
            )
        
        cls_loss = self.cls_loss(cls_logits, target_cls)
        reg_loss = self.reg_loss(bbox_reg, target_reg)
        ctr_loss = self.ctr_loss(centerness, target_ctr)
        
        return cls_loss + reg_loss + ctr_loss

def train_dmy_detector(
    model: DMYDetector,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = "dmy_detector_mps.pth"
) -> Dict[str, Any]:
    """
    Train the DMY detector with MPS support.
    
    Args:
        model: DMYDetector model
        dataloader: Training data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the trained model
        
    Returns:
        Training history
    """
    # Move model to device
    model = model_to_device(model)
    model.train()
    
    # Setup training components
    optimizer = create_optimizer(model, lr=lr)
    scheduler = create_scheduler(optimizer)
    loss_fn = DMYLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'epoch_times': [],
        'memory_usage': []
    }
    
    print(f"Starting DMY detector training on {DEVICE} with dtype {DTYPE}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            # Move data to device
            imgs = to_device(imgs)
            targets = to_device(targets)
            
            # Training step
            try:
                # Forward pass
                cls_logits, bbox_reg, centerness = model(imgs)
                
                # Compute loss
                loss = loss_fn(cls_logits, bbox_reg, centerness, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Progress update
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        memory_info = get_memory_info()
        
        # Update history
        history['train_loss'].append(avg_loss)
        history['epoch_times'].append(epoch_time)
        history['memory_usage'].append(memory_info)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Memory: {memory_info['allocated']:.2f}GB allocated")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
            print(f"  Saved checkpoint: {save_path}_epoch_{epoch+1}.pth")
        
        # Clear memory
        clear_memory()
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\nDMY detector training completed! Final model saved to: {save_path}")
    
    return history

def main():
    """Main training function for DMY detector"""
    print("Initializing DMY Detector Training...")
    
    # Initialize model
    model = DMYDetector()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup data loader
    try:
        dataloader = build_dataloader(root="data/train", batch_size=4)
        print(f"Data loader created with {len(dataloader)} batches")
    except Exception as e:
        print(f"Warning: Could not create data loader: {e}")
        print("Creating dummy data loader for demonstration...")
        dataloader = create_dummy_dataloader(batch_size=4)
    
    # Train the model for 1 epoch
    history = train_dmy_detector(model, dataloader, num_epochs=1)
    
    # Print final statistics
    print("\nTraining Statistics:")
    print(f"Final Loss: {history['train_loss'][-1]:.4f}")
    print(f"Average Epoch Time: {sum(history['epoch_times']) / len(history['epoch_times']):.2f}s")
    print(f"Total Training Time: {sum(history['epoch_times']):.2f}s")

if __name__ == "__main__":
    main() 