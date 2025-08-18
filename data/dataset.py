# data/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import os

class GarmentDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_ids = os.listdir(os.path.join(root, "images"))
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx].split(".")[0]
        # 加载原图、服装mask、人体mask
        img = cv2.imread(os.path.join(self.root, "images", f"{img_id}.jpg"))
        garment_mask = cv2.imread(os.path.join(self.root, "garment_masks", f"{img_id}.png"), 0)
        human_mask = cv2.imread(os.path.join(self.root, "human_masks", f"{img_id}.png"), 0)
        # 加载文本prompt
        with open(os.path.join(self.root, "prompts", f"{img_id}.txt"), "r") as f:
            prompt = f.read()
        # 加载目标服装图像（用于监督训练）
        target_img = cv2.imread(os.path.join(self.root, "target_images", f"{img_id}.jpg"))
        
        # 预处理（resize、归一化等）
        if self.transform:
            img, garment_mask, human_mask, target_img = self.transform(
                img, garment_mask, human_mask, target_img
            )
        return {
            "image": img,
            "garment_mask": garment_mask,
            "human_mask": human_mask,
            "prompt": prompt,
            "target_image": target_img
        }
    
    def __len__(self):
        return len(self.img_ids)