import json
import cv2
import os
import numpy as np
import yaml

def process_human_parsing(image_dir, annos_dir, output_dir, single_sample=False):
    """根据 JSON 文件中的 segmentation 字段处理人体结构 mask"""
    json_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"{annos_folder} 中没有找到 JSON 文件，请检查路径是否正确。")
    
    for json_file in json_files:
        json_path = os.path.join(annos_dir, json_file)
        print(f"正在处理 JSON 文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        
        # 遍历 JSON 文件中的每个标注项
        processed_annotation = False
        for annotation_key, annotation in annos.items():
            img_file_name = f"{json_file.replace('.json', '.jpg')}"  # 假设图像文件名与 JSON 文件名一致
            img_path = os.path.normpath(os.path.join(image_dir, img_file_name))
            if not os.path.exists(img_path):
                print(f"警告：图像 {img_path} 不存在，跳过。")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图像 {img_path}，可能文件已损坏或路径错误，跳过。")
                continue
            
            # 提取 segmentation 信息并生成 mask
            if "segmentation" in annotation and annotation["segmentation"]:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(annotation["segmentation"][0], dtype=np.int32)], 255)
                mask_path = os.path.join(output_dir, "human_masks", f"{json_file.replace('.json', f'_{annotation_key}.png')}")
                # 生成边缘mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                edge_mask = np.zeros_like(mask)
                cv2.drawContours(edge_mask, contours, -1, 255, thickness=5)
                edge_mask_path = os.path.join(output_dir, "edge_masks", f"{json_file.replace('.json', f'_{annotation_key}.png')}")
                cv2.imwrite(edge_mask_path, edge_mask)
                cv2.imwrite(mask_path, mask)
            else:
                print(f"警告：图像 {img_file_name} 的标注项 {annotation_key} 缺少 segmentation 信息，跳过。")
                continue

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(config_path, single_sample=False):
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理 DeepFashion2 数据集
    human_parsing_config = config.get("human_parsing")
    if human_parsing_config:
        output_dir = human_parsing_config.get("output_dir")
        ensure_dir_exists(output_dir)
        
        # 仿照deepfashion2处理不同数据集类型
        datasets = [
            ("train", 
             human_parsing_config.get("train_image_dir"), 
             human_parsing_config.get("train_annos_dir"),
             human_parsing_config.get("train_output_dir")),
            ("validation", 
             human_parsing_config.get("validation_image_dir"), 
             human_parsing_config.get("validation_annos_dir"),
             human_parsing_config.get("validation_output_dir")),
            ("test", 
             human_parsing_config.get("test_image_dir"), 
             human_parsing_config.get("test_annos_dir"),
             human_parsing_config.get("test_output_dir"))
        ]
        
        for dataset_name, image_dir, annos_dir, dataset_output_dir in datasets:
            if not image_dir or not annos_dir:
                print(f"警告：{dataset_name} 数据集配置不完整，跳过处理")
                continue
            
            if not dataset_output_dir:
                dataset_output_dir = os.path.join(output_dir, dataset_name)
                print(f"警告：{dataset_name} 数据集未配置专门的输出目录，使用默认路径: {dataset_output_dir}")
            
            # 创建输出目录结构
            ensure_dir_exists(dataset_output_dir)
            ensure_dir_exists(os.path.join(dataset_output_dir, "human_masks"))
            ensure_dir_exists(os.path.join(dataset_output_dir, "edge_masks"))
            ensure_dir_exists(os.path.join(dataset_output_dir, "images"))
            
            print(f"正在处理 {dataset_name} 数据集，输出到: {dataset_output_dir}")
            process_human_parsing(
                image_dir,
                annos_dir,
                dataset_output_dir,
                single_sample=single_sample
            )
    
    # 处理 DeepFashion2 数据集
    deepfashion2_config = config.get("deepfashion2")
    if deepfashion2_config:
        output_dir = deepfashion2_config.get("output_dir")
        ensure_dir_exists(output_dir)
        ensure_dir_exists(os.path.join(output_dir, "garment_masks"))
        ensure_dir_exists(os.path.join(output_dir, "prompts"))
        
        # 处理 train 数据集
        process_deepfashion2(
            deepfashion2_config.get("train_image_dir"),
            deepfashion2_config.get("train_annos_dir"),
            output_dir,
            single_sample=single_sample
        )
        
        # 处理 validation 数据集
        process_deepfashion2(
            deepfashion2_config.get("validation_image_dir"),
            deepfashion2_config.get("validation_annos_dir"),
            output_dir,
            single_sample=single_sample
        )
        
        # 处理 test 数据集
        process_deepfashion2(
            deepfashion2_config.get("test_image_dir"),
            deepfashion2_config.get("test_annos_dir"),
            output_dir,
            single_sample=single_sample
        )

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess human parsing dataset')
    parser.add_argument('config_path', help='Path to the configuration file')
    parser.add_argument('--single-sample', action='store_true', help='Process only one sample for testing')
    args = parser.parse_args()
    main(args.config_path, single_sample=args.single_sample)