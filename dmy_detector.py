# file: dmy_detector.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from mps_utils import model_to_device, to_device, DEVICE, DTYPE

class DMYHead(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.cls = nn.Conv2d(in_channels, 3, 3, padding=1)   # day, month, year
        self.reg = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.ctn = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, x):
        return self.cls(x), self.reg(x), self.ctn(x)

class DMYDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained backbone with new weights parameter
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        self.head = DMYHead(1024)
        
        # Move to device
        self.to(DEVICE)
        if DEVICE.type == "mps":
            self.to(dtype=DTYPE)

    def forward(self, x):
        # Ensure input is on correct device and dtype
        x = to_device(x)
        
        feat = self.backbone(x)
        return self.head(feat)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with device handling"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint)
        return self