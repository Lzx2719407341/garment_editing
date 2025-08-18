# models/controlnet.py
import torch
import torch.nn as nn
from diffusers import ControlNetModel

class DualBranchControlNet(nn.Module):
    def __init__(self, pretrained_controlnet_path):
        super().__init__()
        # 分支1：服装区域mask编码器（输入单通道mask）
        self.garment_controlnet = ControlNetModel.from_pretrained(
            pretrained_controlnet_path,
            in_channels=1,  # mask为单通道
            low_cpu_mem_usage=True
        )
        # 分支2：人体结构mask编码器（输入单通道mask）
        self.human_controlnet = ControlNetModel.from_pretrained(
            pretrained_controlnet_path,
            in_channels=1,
            low_cpu_mem_usage=True
        )
        # 特征融合层（合并双分支输出）
        self.fusion = nn.Conv2d(
            in_channels=2*320,  # 假设每个ControlNet输出320维特征
            out_channels=320,
            kernel_size=1
        )
    
    def forward(self, x, garment_mask, human_mask, timestep, encoder_hidden_states):
        # 双分支编码
        garment_feat = self.garment_controlnet(
            x, timestep, encoder_hidden_states, garment_mask, return_dict=False
        )[0]  # 取特征部分
        human_feat = self.human_controlnet(
            x, timestep, encoder_hidden_states, human_mask, return_dict=False
        )[0]
        # 融合特征
        fused_feat = self.fusion(torch.cat([garment_feat, human_feat], dim=1))
        return fused_feat