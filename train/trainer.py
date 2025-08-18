# train/trainer.py
import torch
from tqdm import tqdm

class StageTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, config):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = model.device
    
    def train_stage1(self, train_loader, epochs=10):
        """阶段1：仅用服装mask训练，冻结人体结构分支"""
        self.model.controlnet.human_controlnet.requires_grad_(False)  # 冻结人体分支
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()
                # 前向传播（仅用服装mask）
                pred, target = self.model.train_step(batch)
                # 计算损失（忽略人体mask相关项）
                loss = self.loss_fn(
                    pred, target, 
                    garment_mask=batch["garment_mask"],
                    prompt=batch["prompt"]
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            print(f"Stage1 Epoch {epoch}, Loss: {total_loss/len(train_loader)}")
    
    def train_stage2(self, train_loader, epochs=20):
        """阶段2：启用人体结构mask和文本引导"""
        self.model.controlnet.human_controlnet.requires_grad_(True)  # 解冻人体分支
        self.model.train()
        # 训练逻辑类似stage1，但使用完整损失函数
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()
                # 前向传播（使用服装mask和人体mask）
                pred, target = self.model.train_step(batch)
                # 计算损失（包括人体mask相关项）
                loss = self.loss_fn(
                    pred, target, 
                    garment_mask=batch["garment_mask"],
                    human_mask=batch["human_mask"],
                    prompt=batch["prompt"]
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            print(f"Stage2 Epoch {epoch}, Loss: {total_loss/len(train_loader)}")

def train_model(model, dataset, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            images, garment_masks, human_masks, target_images = batch
            # 模型前向传播
            outputs = model(images, garment_masks, human_masks)
            # 计算损失
            loss = loss_fn(outputs, target_images)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()