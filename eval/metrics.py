# eval/metrics.py
import numpy as np
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics import JaccardIndex
import torch
import matplotlib.pyplot as plt

def calculate_mask_iou(pred_mask, gt_mask):
    """计算生成服装区域与真实mask的IOU"""
    jaccard = JaccardIndex(task="binary")
    return jaccard(torch.tensor(pred_mask/255.0), torch.tensor(gt_mask/255.0)).item()

def calculate_fid(generated_images, real_images):
    """计算生成图像与真实图像的FID"""
    fid = FrechetInceptionDistance(feature=64)
    # 预处理图像（归一化到0-1，转为tensor）
    gen_tensor = torch.tensor(np.array(generated_images)/255.0).permute(0, 3, 1, 2).float()
    real_tensor = torch.tensor(np.array(real_images)/255.0).permute(0, 3, 1, 2).float()
    fid.update(gen_tensor, real=False)
    fid.update(real_tensor, real=True)
    return fid.compute().item()

def generate_qualitative_results(original, edited, mask, save_path):
    """生成前后对比图，突出显示编辑区域"""
    # 在原图上叠加mask边界，对比编辑前后效果
    # ...（代码省略，使用matplotlib绘制对比图）
    plt.savefig(save_path)

def evaluate_model(model, dataset):
    fid_scores = []
    iou_scores = []
    for batch in dataset:
        original_images, garment_masks, human_masks, target_images = batch
        edited_images = model(original_images, garment_masks, human_masks)
        # 计算 FID
        fid_scores.append(calculate_fid(edited_images, target_images))
        # 计算 Mask IOU
        iou_scores.append(calculate_mask_iou(garment_masks, edited_images))
    return np.mean(fid_scores), np.mean(iou_scores)