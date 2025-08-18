# models/diffusion_wrapper.py
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

class GarmentDiffusionModel:
    def __init__(self, sd_pretrained_path, controlnet):
        # 加载Stable Diffusion主干
        self.unet = UNet2DConditionModel.from_pretrained(sd_pretrained_path, subfolder="unet")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            sd_pretrained_path,
            unet=self.unet,
            low_cpu_mem_usage=True
        )
        self.controlnet = controlnet
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(self.device)
    
    def train_step(self, batch):
        """训练阶段前向传播"""
        # 准备输入
        x = batch["image"].to(self.device)
        garment_mask = batch["garment_mask"].to(self.device).unsqueeze(1)  # 单通道
        human_mask = batch["human_mask"].to(self.device).unsqueeze(1)
        prompt_embeds = self.pipeline._encode_prompt(
            batch["prompt"], self.device, 1, False
        )
        target = batch["target_image"].to(self.device)
        
        # 随机加噪
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 1000, (x.shape[0],), device=self.device)
        noisy_x = self.pipeline.scheduler.add_noise(x, noise, timesteps)
        
        # ControlNet特征融合
        control_feat = self.controlnet(
            noisy_x, garment_mask, human_mask, timesteps, prompt_embeds
        )
        
        # UNet去噪预测
        model_pred = self.unet(
            noisy_x, timesteps, prompt_embeds, added_cond_kwargs={"text_embeds": control_feat}
        ).sample
        
        return model_pred, target  # 用于计算损失