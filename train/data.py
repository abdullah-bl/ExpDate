# data.py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional
from utils.mps_utils import to_device, DEVICE, DTYPE

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets.
    Ensures all tensors in a batch have the same shape.
    """
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images (they should all be the same size due to transforms)
    images = torch.stack(images, dim=0)
    
    # Handle targets - ensure consistent shapes
    batch_targets = {}
    
    # Get the maximum dimensions for cls, reg, ctr tensors
    max_h, max_w = 0, 0
    for target in targets:
        if 'cls' in target:
            h, w = target['cls'].shape[-2:]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
    
    # If no cls tensors found, use default size
    if max_h == 0 or max_w == 0:
        max_h, max_w = 100, 100
    
    # Create consistent target tensors
    batch_size = len(targets)
    cls_targets = torch.zeros(batch_size, 4, max_h, max_w)
    reg_targets = torch.zeros(batch_size, 4, max_h, max_w)
    ctr_targets = torch.zeros(batch_size, 1, max_h, max_w)
    labels = []
    
    for i, target in enumerate(targets):
        # Copy cls, reg, ctr tensors with padding if needed
        if 'cls' in target:
            cls_tensor = target['cls']
            h, w = cls_tensor.shape[-2:]
            cls_targets[i, :, :h, :w] = cls_tensor
            
        if 'reg' in target:
            reg_tensor = target['reg']
            h, w = reg_tensor.shape[-2:]
            reg_targets[i, :, :h, :w] = reg_tensor
            
        if 'ctr' in target:
            ctr_tensor = target['ctr']
            h, w = ctr_tensor.shape[-2:]
            ctr_targets[i, :, :h, :w] = ctr_tensor
        
        # Store label
        labels.append(target.get('label', {'text': '01/01/2024'}))
    
    batch_targets = {
        'cls': cls_targets,
        'reg': reg_targets,
        'ctr': ctr_targets,
        'labels': labels
    }
    
    return images, batch_targets

class ExpDateDataset(Dataset):
    """Dataset for expiration date recognition with MPS support"""
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing images and annotations
            split: 'train' or 'evaluation'
            transform: Optional transforms to apply
        """
        self.root = root
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        # Load annotations
        annotations_path = os.path.join(root, 'annotations.json')
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            print(f"Warning: No annotations found at {annotations_path}")
            self.annotations = {}
        
        # Get image files
        images_dir = os.path.join(root, 'images')
        if os.path.exists(images_dir):
            self.files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            self.files = [os.path.join(images_dir, f) for f in self.files]
        else:
            print(f"Warning: No images directory found at {images_dir}")
            self.files = []
        
        print(f"Loaded {len(self.files)} images for {split} split")
    
    def _get_default_transform(self):
        """Get default transforms for training/evaluation"""
        if self.split == 'train':
            return T.Compose([
                T.Resize((800, 1333)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((800, 1333)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a single sample"""
        img_path = self.files[idx]
        img_name = os.path.basename(img_path)
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy data
            img = Image.new('RGB', (800, 1333), color='white')
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        # Get annotations
        if img_name in self.annotations:
            label = self.annotations[img_name]
        else:
            # Dummy label for demonstration
            label = {
                'date_bbox': [0, 0, 100, 100],
                'dmy_bboxes': {
                    'day': [0, 0, 50, 50],
                    'month': [50, 0, 100, 50],
                    'year': [0, 50, 100, 100]
                },
                'text': '01/01/2024'
            }
        
        # Create target tensors for training
        target = self._create_target_tensors(label)
        
        return img_tensor, target
    
    def _create_target_tensors(self, label: Dict) -> Dict:
        """Create target tensors for training"""
        # Create consistent target tensors with fixed size
        target_size = (100, 100)  # Fixed size for all targets
        
        # Date detection targets (4 classes: date, due, prod, code)
        cls_target = torch.zeros(4, *target_size)
        reg_target = torch.zeros(4, *target_size)
        ctr_target = torch.zeros(1, *target_size)
        
        # Set some dummy targets in the center
        center_h, center_w = target_size[0] // 2, target_size[1] // 2
        cls_target[0, center_h, center_w] = 1.0  # Date class
        ctr_target[0, center_h, center_w] = 1.0  # Center-ness
        
        return {
            'cls': cls_target,
            'reg': reg_target,
            'ctr': ctr_target,
            'label': label
        }

def build_dataloader(
    root: str = "data/train",
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 0,  # Set to 0 for MPS compatibility
    shuffle: bool = True
) -> DataLoader:
    """
    Build a DataLoader for the ExpDate dataset.
    
    Args:
        root: Dataset root directory
        split: 'train' or 'evaluation'
        batch_size: Batch size
        num_workers: Number of workers (0 for MPS)
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = ExpDateDataset(root, split)
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Disable for MPS
        drop_last=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    return dataloader

def create_dummy_dataloader(batch_size: int = 4, num_samples: int = 100) -> DataLoader:
    """Create a dummy dataloader for testing"""
    class DummyDataset(Dataset):
        def __init__(self, size=num_samples):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Create dummy image and targets with consistent shapes
            img = torch.randn(3, 800, 1333)
            target = {
                'cls': torch.randn(4, 100, 100),
                'reg': torch.randn(4, 100, 100),
                'ctr': torch.randn(1, 100, 100),
                'label': {'text': '01/01/2024'}
            }
            return img, target
    
    dataset = DummyDataset()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )

# Utility functions for data preprocessing
def preprocess_image_for_inference(img_path: str, target_size: Tuple[int, int] = (800, 1333)) -> torch.Tensor:
    """Preprocess a single image for inference"""
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

def postprocess_predictions(cls_logits: List[torch.Tensor], 
                          bbox_reg: List[torch.Tensor], 
                          centerness: List[torch.Tensor],
                          confidence_threshold: float = 0.5) -> List[Dict]:
    """Post-process model predictions to get bounding boxes"""
    predictions = []
    
    for i, (cls, reg, ctr) in enumerate(zip(cls_logits, bbox_reg, centerness)):
        # Apply sigmoid to get probabilities
        cls_probs = torch.sigmoid(cls)
        ctr_probs = torch.sigmoid(ctr)
        
        # Combine classification and centerness
        scores = cls_probs * ctr_probs
        
        # Find high-confidence predictions
        high_conf_mask = scores > confidence_threshold
        
        # Extract bounding boxes (simplified)
        for class_id in range(scores.shape[1]):
            class_scores = scores[:, class_id, :, :]
            class_mask = class_scores > confidence_threshold
            
            if class_mask.any():
                # Get coordinates of high-confidence points
                coords = torch.nonzero(class_mask)
                for coord in coords:
                    b, h, w = coord
                    score = class_scores[b, h, w].item()
                    
                    # Convert to bounding box (simplified)
                    bbox = [w.item() * 8, h.item() * 8, 
                           (w.item() + 1) * 8, (h.item() + 1) * 8]  # Scale factor
                    
                    predictions.append({
                        'bbox': bbox,
                        'class_id': class_id,
                        'score': score,
                        'level': i
                    })
    
    return predictions