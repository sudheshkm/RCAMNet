# dataset.py

from torch.utils.data import Dataset
from PIL import Image
import os

class BLBDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = []
        self.mask_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = sorted(os.listdir(img_dir))

        for cls_idx, cls in enumerate(self.class_names):
            cls_img_dir = os.path.join(img_dir, cls)
            cls_mask_dir = os.path.join(mask_dir, cls)
            if not os.path.isdir(cls_img_dir) or not os.path.isdir(cls_mask_dir):
                continue
            for fname in os.listdir(cls_img_dir):
                if fname.startswith('.') or not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                img_path = os.path.join(cls_img_dir, fname)
                mask_path = os.path.join(cls_mask_dir, fname)
                if not os.path.exists(mask_path):
                    continue
                self.img_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask, label
