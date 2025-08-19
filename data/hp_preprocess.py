import json
import cv2
import os
import numpy as np
import yaml
from PIL import Image

# 定义人体部分颜色映射（来自提供的human_colormap）
HUMAN_COLORMAP = [
    [0, 0, 0],           # 0: 背景
    [128, 0, 0],         # 1: 帽子
    [255, 0, 0],         # 2: 头发
    [0, 85, 0],          # 3: 手套
    [170, 0, 51],        # 4: 太阳镜
    [255, 85, 0],        # 5: 上衣
    [0, 0, 85],          # 6: 裙子
    [0, 119, 221],       # 7: 外套
    [85, 85, 0],         # 8: 袜子
    [0, 85, 85],         # 9: 裤子
    [85, 51, 0],         # 10: 围巾
    [51, 85, 128],       # 11: 脸
    [0, 128, 0],         # 12: 左臂
    [0, 0, 255],         # 13: 右臂
    [51, 170, 221],      # 14: 左腿
    [0, 255, 255],       # 15: 右腿
    [85, 255, 170],      # 16: 左肩
    [170, 255, 85],      # 17: 右肩
    [255, 255, 0],       # 18: 左手
    [255, 170, 0]        # 19: 右手
]

# 非服装区域（需要保留的人体结构）
NON_GARMENT_CLASSES = [2, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 头发、脸、四肢等


def color_to_class_idx(color):
    """将颜色值转换为类别索引"""
    for idx, cmap in enumerate(HUMAN_COLORMAP):
        if np.allclose(color, cmap, atol=1):
            return idx
    return 0  # 默认为背景


def process_human_parsing(image_dir, annos_dir, output_dir, max_images=20000, is_test=False):
    """处理数据集，提取人体结构mask（区分非服装区域），所有数据集（包括测试集）均受max_images限制"""
    # 非测试集必须校验annos_dir，测试集允许annos_dir为空/不存在
    if not is_test:
        if annos_dir is None:
            raise ValueError("annos_dir参数不能为空，请检查配置文件是否正确设置")
        if not os.path.isdir(annos_dir):
            raise NotADirectoryError(f"annos_dir={annos_dir}不是有效的目录路径")
    
    # 获取图像ID列表（测试集无annos_dir时从image_dir提取）
    if is_test and (annos_dir is None or not os.path.isdir(annos_dir)):
        # 测试集无标注目录：从image_dir读取图像文件名作为ID
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"测试集图像目录不存在: {image_dir}")
        img_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        img_ids = [os.path.splitext(f)[0] for f in img_files if f.strip()]
        if not img_ids:
            raise FileNotFoundError(f"测试集图像目录 {image_dir} 中未找到有效图像")
    else:
        # 有标注目录（包括非测试集和有标注的测试集）：从id_file读取ID
        id_file = os.path.join(annos_dir, 
                              "train_id.txt" if "Training" in annos_dir else 
                              "val_id.txt" if "Validation" in annos_dir else 
                              "test_id.txt")
        if not os.path.exists(id_file):
            raise FileNotFoundError(f"{id_file} 中没有找到ID文件，请检查路径是否正确。")
        with open(id_file, 'r') as f:
            img_ids = [line.strip() for line in f.readlines() if line.strip()]
        if not img_ids:
            raise FileNotFoundError(f"{id_file} 中未包含有效图像ID，请检查文件内容。")
    
    processed_count = 0
    print(f"开始处理 {'测试集（无标注）' if (is_test and (annos_dir is None or not os.path.isdir(annos_dir))) else '数据集'}: {image_dir}")
    print(f"当前数据集最大处理数量限制: {max_images if max_images is not None else '无限制'}")
    
    for img_id in img_ids:
        # 读取原始图像
        img_file_name = f"{img_id}.jpg"  # 假设图像为jpg格式，可扩展其他格式
        img_path = os.path.normpath(os.path.join(image_dir, img_file_name))
        if not os.path.exists(img_path):
            print(f"警告：图像 {img_path} 不存在，跳过。")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过。")
            continue

        # 保存原始图像（无论是否有标注，都保留原始图像）
        img_save_path = os.path.normpath(os.path.join(output_dir, "images", img_file_name))
        ensure_dir_exists(os.path.dirname(img_save_path))  # 确保目录存在
        cv2.imwrite(img_save_path, img)
        print(f"已保存原始图像: {img_save_path}")

        # 测试集无标注目录时，不生成mask，直接跳过
        if is_test and (annos_dir is None or not os.path.isdir(annos_dir)):
            processed_count += 1
            print(f"测试集无标注，仅保存原始图像（已处理 {processed_count} 个样本）")
        else:
            # 以下为有标注时的mask生成逻辑（非测试集或有标注的测试集）
            h, w = img.shape[:2]
            human_mask = np.zeros((h, w), dtype=np.uint8)

            if not is_test:
                # 非测试集：基于标注生成mask
                category_file_name = f"{img_id}.png"
                category_path = os.path.normpath(os.path.join(annos_dir, "Categories", category_file_name))
                if not os.path.exists(category_path):
                    print(f"警告：类别标注 {category_path} 不存在，跳过。")
                    continue
                category_img = cv2.imread(category_path)
                if category_img is None:
                    print(f"警告：无法读取类别标注 {category_path}，跳过。")
                    continue
                category_img = cv2.cvtColor(category_img, cv2.COLOR_BGR2RGB)

                # 生成非服装区域mask
                for i in range(h):
                    for j in range(w):
                        color = category_img[i, j]
                        class_idx = color_to_class_idx(color)
                        if class_idx in NON_GARMENT_CLASSES:
                            human_mask[i, j] = 255

                # 过滤过小mask
                mask_area = np.sum(human_mask > 0)
                image_area = h * w
                if mask_area < image_area * 0.005:
                    print(f"警告：图像 {img_file_name} 的人体mask面积过小，跳过。")
                    continue
            else:
                # 测试集有标注目录时（特殊情况）：按原逻辑生成默认mask
                human_mask = np.ones((h, w), dtype=np.uint8) * 255
                print(f"测试集有标注目录，生成默认全图人体mask")

            # 保存human_mask和edge_mask（仅在有标注时执行）
            mask_path = os.path.normpath(os.path.join(output_dir, "human_masks", f"{img_id}.png"))
            ensure_dir_exists(os.path.dirname(mask_path))
            cv2.imwrite(mask_path, human_mask)
            print(f"已保存人体mask: {mask_path}")

            contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_mask = np.zeros_like(human_mask)
            cv2.drawContours(edge_mask, contours, -1, 255, thickness=3)
            edge_mask_path = os.path.normpath(os.path.join(output_dir, "edge_masks", f"{img_id}.png"))
            ensure_dir_exists(os.path.dirname(edge_mask_path))
            cv2.imwrite(edge_mask_path, edge_mask)
            print(f"已保存边缘mask: {edge_mask_path}")

            processed_count += 1
            print(f"已处理 {processed_count} 个样本")

        # 所有数据集（包括测试集）均受max_images限制，达到上限则停止
        if max_images is not None and processed_count >= max_images:
            print(f"已达到最大处理数量 {max_images}，停止处理当前数据集")
            break

    print(f"处理完成，共处理 {processed_count} 个样本，输出目录: {output_dir}")


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(config_path):
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    human_parsing_config = config.get("human_parsing")
    if not human_parsing_config:
        raise ValueError("配置文件中未找到human_parsing配置，请检查配置文件是否正确设置")

    # 从配置中读取最大处理图片数量（全局生效，包括测试集）
    max_images = human_parsing_config.get("max_images", None)
    if max_images is not None:
        print(f"全局最大处理样本数量设置为: {max_images}")

    output_dir = human_parsing_config.get("output_dir")
    ensure_dir_exists(output_dir)

    # 处理训练集、验证集、测试集（均受max_images限制）
    datasets = [
        ("train", 
         human_parsing_config.get("train_image_dir"), 
         human_parsing_config.get("train_annos_dir"),
         human_parsing_config.get("train_output_dir"),
         False),  # 非测试集
        ("validation", 
         human_parsing_config.get("validation_image_dir"), 
         human_parsing_config.get("validation_annos_dir"),
         human_parsing_config.get("validation_output_dir"),
         False),  # 非测试集
        ("test", 
         human_parsing_config.get("test_image_dir"), 
         human_parsing_config.get("test_annos_dir"),
         human_parsing_config.get("test_output_dir"),
         True)    # 测试集
    ]

    for dataset_name, image_dir, annos_dir, dataset_output_dir, is_test in datasets:
        # 校验图像目录是否存在（所有数据集都必须有图像目录）
        if not image_dir or not os.path.isdir(image_dir):
            print(f"警告：{dataset_name} 数据集图像目录不存在 - {image_dir}，跳过处理")
            continue
        
        # 非测试集必须校验标注目录，测试集允许标注目录不存在
        if not is_test and (not annos_dir or not os.path.isdir(annos_dir)):
            print(f"警告：{dataset_name} 数据集标注目录不存在 - {annos_dir}，跳过处理")
            continue

        # 输出目录处理
        if not dataset_output_dir:
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            print(f"警告：{dataset_name} 数据集未配置专门的输出目录，使用默认路径: {dataset_output_dir}")
        
        # 创建输出目录结构
        ensure_dir_exists(dataset_output_dir)
        ensure_dir_exists(os.path.join(dataset_output_dir, "human_masks"))
        ensure_dir_exists(os.path.join(dataset_output_dir, "edge_masks"))
        ensure_dir_exists(os.path.join(dataset_output_dir, "images"))
        
        print(f"正在处理 {dataset_name} 数据集，输出到: {dataset_output_dir}")
        
        # 处理数据集（测试集和其他数据集均受max_images限制）
        process_human_parsing(
            image_dir=image_dir,
            annos_dir=annos_dir,
            output_dir=dataset_output_dir,
            max_images=max_images,
            is_test=is_test
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hp_preprocess.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])