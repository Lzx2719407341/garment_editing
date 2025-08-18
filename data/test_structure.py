#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from test_preprocess import main

def test_directory_structure():
    """测试修改后的代码是否创建了正确的目录结构"""
    config_path = "../configs/data_config.yaml"
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['deepfashion2']['output_dir']
    
    print("测试目录结构...")
    print(f"基础输出目录: {output_dir}")
    
    # 期望的目录结构
    expected_dirs = [
        os.path.join(output_dir, "train", "garment_masks"),
        os.path.join(output_dir, "train", "prompts"),
        os.path.join(output_dir, "validation", "garment_masks"), 
        os.path.join(output_dir, "validation", "prompts"),
        os.path.join(output_dir, "test", "garment_masks"),
        os.path.join(output_dir, "test", "prompts"),
    ]
    
    print("\n期望的目录结构:")
    for dir_path in expected_dirs:
        print(f"  {dir_path}")
    
    # 检查当前实际存在的目录
    print(f"\n当前 {output_dir} 下的实际结构:")
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"  输出目录 {output_dir} 不存在")

if __name__ == "__main__":
    test_directory_structure()
