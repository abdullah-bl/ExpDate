# file: dan_recognizer.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet50, ResNet50_Weights
from mps_utils import model_to_device, to_device, DEVICE, DTYPE

class DAN(nn.Module):
    """
    Very small footprint of Decoupled Attention Network (DAN).
    Optimized for MPS with proper device handling.
    """
    def __init__(self, vocab_size=37, max_len=20, d_model=512):
        super().__init__()
        # Load pretrained backbone with new weights parameter
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # Positional encoding
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True  # Better for MPS
        )
        self.trans = TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Move to device
        self.to(DEVICE)
        if DEVICE.type == "mps":
            self.to(dtype=DTYPE)

    def forward(self, x):
        # Ensure input is on correct device and dtype
        x = to_device(x)
        
        # Extract features from backbone
        feat = self.backbone(x)           # (B, 2048)
        
        # Project to d_model if needed
        if feat.shape[-1] != self.pos.shape[-1]:
            feat = nn.functional.linear(feat, 
                                      torch.randn(self.pos.shape[-1], feat.shape[-1]).to(DEVICE))
        
        # Add positional encoding
        feat = feat.unsqueeze(1).repeat(1, self.pos.shape[1], 1) + self.pos
        
        # Transformer processing
        out = self.trans(feat)           # (B, 20, d_model)
        
        # Output projection
        return self.fc(out)              # (B, 20, vocab)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with device handling"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint)
        return self