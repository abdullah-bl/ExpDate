# file: date_detector.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from mps_utils import model_to_device, to_device, DEVICE, DTYPE

# Fallback FPN implementation if mmdet is not available
class SimpleFPN(nn.Module):
    """Simple FPN implementation as fallback"""
    def __init__(self, in_channels_list, out_channels, num_outs):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(num_outs):
            lateral_conv = nn.Conv2d(in_channels_list[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs):
        # inputs should be [c2, c3, c4, c5]
        laterals = [self.lateral_convs[i](inputs[i]) for i in range(len(inputs))]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += nn.functional.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest'
            )
        
        # FPN convs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs

class DateHead(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.cls = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.ctn = nn.Conv2d(in_channels, 1, 3, padding=1)   # center-ness

    def forward(self, feats):
        cls_logits, bbox_reg, centerness = [], [], []
        for f in feats:
            cls_logits.append(self.cls(f))
            bbox_reg.append(self.reg(f))
            centerness.append(self.ctn(f))
        return cls_logits, bbox_reg, centerness

class DateDetector(nn.Module):
    def __init__(self, use_mmdet_fpn=True):
        super().__init__()
        # Load pretrained backbone with new weights parameter
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        
        # FPN setup with fallback
        try:
            if use_mmdet_fpn:
                from mmdet.models.necks import FPN
                self.fpn = FPN([256, 512, 1024, 2048], 256, 4)
            else:
                raise ImportError("Using fallback FPN")
        except ImportError:
            print("Warning: mmdet not available, using fallback FPN")
            self.fpn = SimpleFPN([256, 512, 1024, 2048], 256, 4)
        
        self.head = DateHead(256, num_classes=4)  # date, due, prod, code
        
        # Move to device
        self.to(DEVICE)
        if DEVICE.type == "mps":
            self.to(dtype=DTYPE)

    def forward(self, x):
        # Ensure input is on correct device and dtype
        x = to_device(x)
        
        # Extract features from backbone
        features = []
        x_input = x
        for i, layer in enumerate(self.body):
            x_input = layer(x_input)
            if i in [4, 5, 6, 7]:  # c2, c3, c4, c5
                features.append(x_input)
        
        # FPN processing
        feats = self.fpn(features)
        
        # Head processing
        return self.head(feats)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with device handling"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint)
        return self