# models/loss.py
import torch
import torch.nn as nn
from lpips import LPIPS

class GarmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").eval()  # 感知损失
        self.clip_model = torch.hub.load("openai/CLIP", "ViT-L/14")  # 语义损失
    
    def forward(self, pred, target, garment_mask, prompt):
        # 1. 重建损失（仅在服装区域计算）
        mask = garment_mask / 255.0  # 归一化到0-1
        l1 = self.l1_loss(pred * mask, target * mask)
        
        # 2. 感知损失
        lpips = self.lpips_loss(pred, target).mean()
        
        # 3. mask IOU损失（最大化生成区域与mask的重叠）
        pred_mask = (pred > 0.5).float()  # 假设二值化
        intersection = (pred_mask * mask).sum()
        union = (pred_mask + mask).sum() - intersection
        iou_loss = 1 - (intersection + 1e-8) / (union + 1e-8)
        
        # 4. 文本语义损失（CLIP相似度）
        pred_feat = self.clip_model.encode_image(pred)
        text_feat = self.clip_model.encode_text(prompt)
        clip_sim = torch.cosine_similarity(pred_feat, text_feat).mean()
        clip_loss = 1 - clip_sim
        
        # 总损失（加权求和）
        total_loss = 0.5*l1 + 0.3*lpips + 0.1*iou_loss + 0.1*clip_loss
        return total_loss