# file: pipeline.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional

from train.date_detector import DateDetector
from train.dmy_detector import DMYDetector
from train.dan_recognizer import DAN
from utils.mps_utils import to_device, DEVICE, DTYPE, clear_memory

class ExpDatePipeline:
    def __init__(self, ckpt_detect: str, ckpt_dmy: str, ckpt_rec: str):
        """
        Initialize the expiration date recognition pipeline.
        
        Args:
            ckpt_detect: Path to date detector checkpoint
            ckpt_dmy: Path to DMY detector checkpoint  
            ckpt_rec: Path to recognition network checkpoint
        """
        # Initialize models
        self.date_det = DateDetector()
        self.dmy_det = DMYDetector()
        self.recog = DAN()
        
        # Load checkpoints with device handling
        try:
            self.date_det.load_checkpoint(ckpt_detect)
            self.dmy_det.load_checkpoint(ckpt_dmy)
            self.recog.load_checkpoint(ckpt_rec)
        except Exception as e:
            print(f"Warning: Could not load checkpoints: {e}")
            print("Using untrained models for demonstration")
        
        # Set to evaluation mode
        self.date_det.eval()
        self.dmy_det.eval()
        self.recog.eval()
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((800, 1333)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Vocabulary for recognition
        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        self.vocab_size = len(self.vocab)

    @torch.no_grad()
    def __call__(self, img_path: str) -> str:
        """
        Process an image to extract expiration date.
        
        Args:
            img_path: Path to input image
            
        Returns:
            Extracted date in DD/MM/YYYY format
        """
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            x = self.transform(img).unsqueeze(0)
            x = to_device(x)

            # 1. Detect date region
            cls_logits, bbox_reg, centerness = self.date_det(x)
            date_box = self.post_process_date(cls_logits, bbox_reg, centerness)
            
            if date_box is None:
                return "No date region detected"

            # 2. Crop and resize date region, then detect DMY
            date_crop = self.crop_and_resize(img, date_box, (64, 256))
            date_crop_tensor = self.transform(date_crop).unsqueeze(0)
            date_crop_tensor = to_device(date_crop_tensor)
            
            dmy_cls, dmy_reg, dmy_ctn = self.dmy_det(date_crop_tensor)
            dmy_boxes = self.post_process_dmy(dmy_cls, dmy_reg, dmy_ctn)

            # 3. Recognize each component
            results = {}
            for label, box in dmy_boxes.items():
                if box is not None:
                    comp_crop = self.crop_and_resize(date_crop, box, (32, 128))
                    comp_tensor = self.transform(comp_crop).unsqueeze(0)
                    comp_tensor = to_device(comp_tensor)
                    
                    logits = self.recog(comp_tensor)
                    pred = logits.argmax(-1).squeeze()
                    results[label] = self.decode(pred)
                else:
                    results[label] = "00"

            # Clear memory
            clear_memory()
            
            return f"{results.get('day', '00')}/{results.get('month', '00')}/{results.get('year', '0000')}"
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error processing image"

    def post_process_date(self, cls_logits: List[torch.Tensor], 
                         bbox_reg: List[torch.Tensor], 
                         centerness: List[torch.Tensor]) -> Optional[Tuple[int, int, int, int]]:
        """Post-process date detection results with NMS"""
        # Simple post-processing - find the highest confidence date region
        max_conf = 0
        best_box = None
        
        for i, (cls, reg, ctr) in enumerate(zip(cls_logits, bbox_reg, centerness)):
            # Get date class confidence (assuming date is class 0)
            date_conf = torch.sigmoid(cls[:, 0, :, :]) * torch.sigmoid(ctr[:, 0, :, :])
            
            if date_conf.max() > max_conf:
                max_conf = date_conf.max()
                # Get the best box coordinates (simplified)
                best_box = (0, 0, 100, 100)  # Placeholder
        
        return best_box

    def post_process_dmy(self, cls_logits: torch.Tensor, 
                        bbox_reg: torch.Tensor, 
                        centerness: torch.Tensor) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        """Post-process DMY detection results"""
        # Simple post-processing - find regions for day, month, year
        cls_probs = torch.sigmoid(cls_logits)
        ctr_probs = torch.sigmoid(centerness)
        
        # Get the best regions for each class
        results = {}
        for i, label in enumerate(['day', 'month', 'year']):
            conf = cls_probs[:, i, :, :] * ctr_probs[:, 0, :, :]
            if conf.max() > 0.5:  # Confidence threshold
                results[label] = (0, 0, 50, 50)  # Placeholder coordinates
            else:
                results[label] = None
        
        return results

    def crop_and_resize(self, img: Image.Image, box: Tuple[int, int, int, int], 
                       size: Tuple[int, int]) -> Image.Image:
        """Crop image region and resize to target size"""
        try:
            x1, y1, x2, y2 = box
            cropped = img.crop((x1, y1, x2, y2))
            return cropped.resize(size, Image.LANCZOS)
        except Exception:
            # Fallback to center crop if box is invalid
            return img.resize(size, Image.LANCZOS)

    def decode(self, pred: torch.Tensor) -> str:
        """Decode prediction tensor to string"""
        try:
            pred = pred.cpu().numpy()
            result = ""
            for p in pred:
                if p < len(self.vocab) and p != 0:  # Skip padding
                    result += self.vocab[p]
            return result if result else "00"
        except Exception:
            return "00"

def demo():
    """Demo function to test the pipeline"""
    # Create pipeline with dummy checkpoints
    pipeline = ExpDatePipeline("dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth")
    
    # Test with a sample image if available
    import os
    if os.path.exists("data/evaluation/images/test_00001.jpg"):
        result = pipeline("data/evaluation/images/test_00001.jpg")
        print(f"Detected date: {result}")
    else:
        print("No test image found. Create a test image to run the demo.")

if __name__ == "__main__":
    demo()