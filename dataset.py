# dataset.py
import torch, json, cv2, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pathlib import Path

class ExpDateDataset(Dataset):
    """
    Loads:
        data/
        ├── train/annotations.json
        ├── train/img_00001.jpg
        ├── train/img_00002.jpg
        └── ...
    """
    CLASS2ID = {"date": 0, "due": 1, "prod": 2, "code": 3}

    def __init__(self,
                 root: str,
                 split: str = "train",
                 img_size: tuple = (800, 1333)):
        self.root = Path(root) / split
        self.size = img_size
        with open(self.root / "annotations.json") as f:
            self.meta = json.load(f)
        self.ids = sorted(self.meta.keys())

    def __len__(self):
        return len(self.ids)

    def _load_target(self, img_id):
        """
        Returns torch tensors ready for torchvision detection models.
        """
        ann_list = self.meta[img_id]["ann"]
        boxes, labels = [], []
        for a in ann_list:
            x_min, y_min, x_max, y_max = a["bbox"]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.CLASS2ID[a["cls"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
        }

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.root / img_id
        img = cv2.imread(str(img_path))[:, :, ::-1]  # BGR→RGB
        img = cv2.resize(img, self.size[::-1])        # (W,H)

        target = self._load_target(img_id)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target