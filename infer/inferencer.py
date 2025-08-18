# infer/inferencer.py
import torch
import cv2
import numpy as np
from diffusers import UniPCMultistepScheduler

class GarmentInferencer:
    def __init__(self, model_path, controlnet_path):
        # 加载训练好的模型
        self.model = GarmentDiffusionModel(
            sd_pretrained_path=model_path,
            controlnet=DualBranchControlNet(controlnet_path)
        )
        self.model.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.model.pipeline.scheduler.config
        )
        self.model.eval()
    
    def edit_garment(self, input_img, garment_mask, prompt, human_mask=None):
        """
        输入：
            input_img: 原始图像（cv2格式）
            garment_mask: 待编辑区域mask（单通道）
            prompt: 文本描述（如"替换为黑色西装"）
            human_mask: 人体结构mask（可选，自动生成若为None）
        输出：
            edited_img: 编辑后的图像
        """
        # 预处理输入
        if human_mask is None:
            human_mask = self._auto_generate_human_mask(input_img)  # 调用人体解析模型生成
        img_tensor = self._preprocess(input_img).unsqueeze(0).to(self.model.device)
        garment_mask_tensor = self._preprocess(garment_mask, is_mask=True).unsqueeze(0).to(self.model.device)
        human_mask_tensor = self._preprocess(human_mask, is_mask=True).unsqueeze(0).to(self.model.device)
        
        # 推理生成
        with torch.no_grad():
            edited_img = self.model.pipeline(
                prompt=prompt,
                image=img_tensor,
                controlnet_cond=torch.cat([garment_mask_tensor, human_mask_tensor], dim=1),
                num_inference_steps=50,
                guidance_scale=7.5,
                controlnet_conditioning_scale=[1.0, 0.8]  # 两个分支的控制强度
            ).images[0]
        
        return cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)
    
    def _auto_generate_human_mask(self, img):
        """调用预训练人体解析模型生成人体结构mask"""
        # 示例：使用预训练的HRNet模型
        # ...（代码省略，需集成人体解析模型）
        return human_mask