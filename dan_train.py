# dan_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any

from dan_recognizer import DAN
from data import build_dataloader, create_dummy_dataloader
from mps_utils import (
    model_to_device, to_device, create_optimizer, create_scheduler,
    MixedPrecisionTrainer, clear_memory, get_memory_info, DEVICE, DTYPE
)

class DANLoss(nn.Module):
    """Loss function for DAN text recognition"""
    def __init__(self, vocab_size=37, ignore_index=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.vocab_size = vocab_size
    
    def forward(self, logits, targets):
        """
        Compute DAN loss.
        
        Args:
            logits: (B, seq_len, vocab_size) - model predictions
            targets: (B, seq_len) - target sequences
        """
        # Reshape for CrossEntropyLoss: (B*seq_len, vocab_size)
        B, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(B * seq_len, vocab_size)
        targets_flat = targets.view(B * seq_len)
        
        return self.criterion(logits_flat, targets_flat)

def create_text_targets(texts: list, max_len: int = 20, vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-") -> torch.Tensor:
    """Create target tensors for text recognition"""
    vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # 0 is padding
    vocab_dict['<PAD>'] = 0
    
    targets = []
    for text in texts:
        # Convert text to indices
        indices = [vocab_dict.get(char, 0) for char in text[:max_len]]
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)
        
        targets.append(indices)
    
    return torch.tensor(targets, dtype=torch.long)

def train_dan(
    model: DAN,
    dataloader: DataLoader,
    num_epochs: int = 15,
    lr: float = 1e-4,
    save_path: str = "dan_recognizer_mps.pth"
) -> Dict[str, Any]:
    """
    Train the DAN recognizer with MPS support.
    
    Args:
        model: DAN model
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
    optimizer = create_optimizer(model, lr=lr, weight_decay=1e-5)
    scheduler = create_scheduler(optimizer, step_size=5, gamma=0.5)
    loss_fn = DANLoss()
    
    # Mixed precision trainer
    trainer = MixedPrecisionTrainer(model)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'epoch_times': [],
        'memory_usage': []
    }
    
    print(f"Starting DAN training on {DEVICE} with dtype {DTYPE}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            # Move data to device
            imgs = to_device(imgs)
            
            # Create text targets from labels
            texts = [target['label']['text'] for target in targets]
            text_targets = create_text_targets(texts)
            text_targets = to_device(text_targets)
            
            # Training step
            try:
                # Forward pass
                logits = model(imgs)
                
                # Compute loss
                loss = loss_fn(logits, text_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Compute accuracy
                pred = logits.argmax(-1)
                acc = (pred == text_targets).float().mean().item()
                
                epoch_loss += loss.item()
                epoch_acc += acc
                num_batches += 1
                
                # Progress update
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        memory_info = get_memory_info()
        
        # Update history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['epoch_times'].append(epoch_time)
        history['memory_usage'].append(memory_info)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_acc:.4f}")
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
    print(f"\nDAN training completed! Final model saved to: {save_path}")
    
    return history

def main():
    """Main training function for DAN"""
    print("Initializing DAN Training...")
    
    # Initialize model
    model = DAN()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup data loader
    try:
        dataloader = build_dataloader(root="data/train", batch_size=8)
        print(f"Data loader created with {len(dataloader)} batches")
    except Exception as e:
        print(f"Warning: Could not create data loader: {e}")
        print("Creating dummy data loader for demonstration...")
        dataloader = create_dummy_dataloader(batch_size=8)
    
    # Train the model
    history = train_dan(model, dataloader)
    
    # Print final statistics
    print("\nTraining Statistics:")
    print(f"Final Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Average Epoch Time: {sum(history['epoch_times']) / len(history['epoch_times']):.2f}s")
    print(f"Total Training Time: {sum(history['epoch_times']):.2f}s")

if __name__ == "__main__":
    main() 