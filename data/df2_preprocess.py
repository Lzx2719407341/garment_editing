import json
import cv2
import os
import numpy as np
import yaml


def process_deepfashion2(image_dir, annos_dir, output_dir, single_sample=False, max_images=20000):
    """提取服装 mask 和属性文本，添加single_sample参数控制是否只处理一个数据实例"""
    # 验证annos_dir参数有效性
    if annos_dir is None:
        raise ValueError("annos_dir参数不能为空，请检查配置文件是否正确设置")
    if not os.path.isdir(annos_dir):
        raise NotADirectoryError(f"annos_dir={annos_dir}不是有效的目录路径")
        
    json_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"{annos_dir} 中没有找到 JSON 文件，请检查路径是否正确。")
    
    # 如果只处理单个样本，只取第一个JSON文件
    if single_sample:
        json_files = [json_files[0]]
        print(f"启用单样本测试模式，仅处理文件: {json_files[0]}")
    
    # 添加处理图片计数器
    processed_images_count = 0
    
    for json_file in json_files:
            
        json_path = os.path.join(annos_dir, json_file)
        print(f"正在处理 JSON 文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        print(f"JSON 文件内容: {annos.keys()}")
        
        # 遍历 JSON 文件中的每个标注项
        processed_annotation = False
        for item_key, annotation in annos.items():
            if item_key == "source" or item_key == "pair_id":
                continue  # 跳过非标注字段
            
            # 如果只处理单个样本，处理完第一个有效标注项后跳出循环
            if single_sample and processed_annotation:
                break
            
            img_file_name = f"{json_file.replace('.json', '.jpg')}"  # 假设图像文件名与 JSON 文件名一致
            img_path = os.path.normpath(os.path.join(image_dir, img_file_name))
            if not os.path.exists(img_path):
                print(f"警告：图像 {img_path} 不存在，跳过。")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图像 {img_path}，可能文件已损坏或路径错误，跳过。")
                continue
            
            # 提取服装 mask
            if "segmentation" in annotation and annotation["segmentation"]:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for segment in annotation["segmentation"]:
                    # 转换 segment 为二维数组
                    segment = np.array(segment, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [segment], 255)
                
                # 添加mask质量控制：过滤掉面积过小的mask
                mask_area = np.sum(mask > 0)
                image_area = img.shape[0] * img.shape[1]
                if mask_area < image_area * 0.01:  # mask面积小于图像面积1%则跳过
                    print(f"警告：图像 {img_file_name} 的mask面积过小，跳过。")
                    continue
                
                mask_path = os.path.join(output_dir, "garment_masks", f"{json_file.replace('.json', f'_{item_key}.png')}")
                print(f"正在保存 mask 文件: {mask_path}")
                cv2.imwrite(mask_path, mask)
                
                # 添加mask质量控制：过滤掉面积过小的mask
                mask_area = np.sum(mask > 0)
                image_area = img.shape[0] * img.shape[1]
                if mask_area < image_area * 0.01:  # mask面积小于图像面积1%则跳过
                    print(f"警告：图像 {img_file_name} 的mask面积过小，跳过。")
                    continue
                
                mask_path = os.path.join(output_dir, "garment_masks", f"{json_file.replace('.json', f'_{item_key}.png')}")
                print(f"正在保存 mask 文件: {mask_path}")
                cv2.imwrite(mask_path, mask)
                
                # 生成边缘mask用于精确控制
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                edge_mask = np.zeros_like(mask)
                cv2.drawContours(edge_mask, contours, -1, 255, thickness=5)  # 绘制5像素宽度的边缘
                edge_mask_path = os.path.join(output_dir, "edge_masks", f"{json_file.replace('.json', f'_{item_key}.png')}")
                cv2.imwrite(edge_mask_path, edge_mask)
                
                
            else:
                print(f"警告：图像 {img_file_name} 的标注项 {item_key} 缺少 segmentation 信息，跳过。")
                continue
            
            # 转换属性为文本 prompt
            attributes = annotation.get("attributes", {})
            prompt = []
            if "color" in attributes:
                prompt.append(attributes["color"])
            if "category_name" in annotation:
                prompt.append(annotation["category_name"])
            if "style" in annotation:
                prompt.append(f"style {annotation['style']}")
            prompt_text = " ".join(prompt)
            
            # 优化文本提示：添加更详细的描述
            if "occlusion" in annotation:
                if annotation["occlusion"] == 1:
                    prompt_text += " slightly occluded"
                elif annotation["occlusion"] == 2:
                    prompt_text += " heavily occluded"
            
            prompt_path = os.path.join(output_dir, "prompts", f"{json_file.replace('.json', f'_{item_key}.txt')}")
            print(f"正在保存 prompt 文件: {prompt_path}")
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            
            # 保存原始图像
            original_img_path = os.path.join(output_dir, "images", f"{json_file.replace('.json', f'_{item_key}.jpg')}")
            cv2.imwrite(original_img_path, img)
            
            # 保存bounding box信息
            if "bounding_box" in annotation:
                bounding_box = annotation["bounding_box"]
                bounding_box_path = os.path.join(output_dir, "bounding_boxes", f"{json_file.replace('.json', f'_{item_key}.txt')}")
                with open(bounding_box_path, "w") as f:
                    f.write(f"{bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}")  # x, y, width, height
            
            # 保存关键点信息
            if "landmarks" in annotation:
                landmarks = annotation["landmarks"]
                landmarks_path = os.path.join(output_dir, "landmarks", f"{json_file.replace('.json', f'_{item_key}.json')}")
                with open(landmarks_path, "w") as f:
                    json.dump(landmarks, f)
            
            # 保存类别信息
            if "category_name" in annotation:
                category = annotation["category_name"]
                category_path = os.path.join(output_dir, "categories", f"{json_file.replace('.json', f'_{item_key}.txt')}")
                with open(category_path, "w") as f:
                    f.write(category)
            
            # 标记已处理一个标注项
            processed_annotation = True
            
        # 增加处理图片计数
        processed_images_count += 1
            
        # 如果设置了最大处理图片数，且已达到限制，则停止处理
        if max_images is not None and processed_images_count >= max_images:
                print(f"已处理 {processed_images_count} 张图片，达到设置的最大处理数量 {max_images}，停止处理。")
                break
        
        # 如果只处理单个样本，处理完一个文件后跳出循环
        if single_sample:
            break
        

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(config_path):
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理 DeepFashion2 数据集
    deepfashion2_config = config.get("deepfashion2")
    if deepfashion2_config:
        output_dir = deepfashion2_config.get("output_dir")
        ensure_dir_exists(output_dir)
        
        # 从配置中读取最大处理图片数量
        max_images = deepfashion2_config.get("max_images", None)
        if max_images is not None:
            print(f"设置最大处理图片数量为: {max_images}")
        
        # 为不同数据集创建单独的输出目录，从配置文件中读取专门的输出路径
        datasets = [
            ("train", 
             deepfashion2_config.get("train_image_dir"), 
             deepfashion2_config.get("train_annos_dir"),
             deepfashion2_config.get("train_output_dir")),
            ("validation", 
             deepfashion2_config.get("validation_image_dir"), 
             deepfashion2_config.get("validation_annos_dir"),
             deepfashion2_config.get("validation_output_dir")),
            ("test", 
             deepfashion2_config.get("test_image_dir"), 
             deepfashion2_config.get("test_annos_dir"),
             deepfashion2_config.get("test_output_dir"))
        ]
        
        for dataset_name, image_dir, annos_dir, dataset_output_dir in datasets:
            if not image_dir or not annos_dir:
                print(f"警告：{dataset_name} 数据集配置不完整，跳过处理")
                continue
            
            # 验证目录是否存在
            if not (os.path.isdir(image_dir) and os.path.isdir(annos_dir)):
                print(f"警告：{dataset_name} 数据集路径不存在 - 图像目录: {image_dir}, 标注目录: {annos_dir}")
                continue
            
            # 如果配置文件中没有指定专门的输出目录，则回退到原有逻辑
            if not dataset_output_dir:
                dataset_output_dir = os.path.join(output_dir, dataset_name)
                print(f"警告：{dataset_name} 数据集未配置专门的输出目录，使用默认路径: {dataset_output_dir}")
            
            # 创建输出目录结构
            ensure_dir_exists(dataset_output_dir)
            # 保存服装mask的目录 - 用于存储提取出的服装区域掩码
            ensure_dir_exists(os.path.join(dataset_output_dir, "garment_masks"))
            # 保存文本提示的目录 - 用于存储与服装相关的文本描述
            ensure_dir_exists(os.path.join(dataset_output_dir, "prompts"))
            # 保存边缘mask的目录 - 用于存储服装边缘信息，便于精确编辑
            ensure_dir_exists(os.path.join(dataset_output_dir, "edge_masks"))
            # 保存图像的目录 - 用于存储原始图像及数据增强后的图像
            ensure_dir_exists(os.path.join(dataset_output_dir, "images"))
            # 保存边界框的目录 - 用于存储服装的边界框坐标信息
            ensure_dir_exists(os.path.join(dataset_output_dir, "bounding_boxes"))
            # 保存关键点的目录 - 用于存储服装的关键点标注信息
            ensure_dir_exists(os.path.join(dataset_output_dir, "landmarks"))
            # 保存类别信息的目录 - 用于存储服装的分类信息
            ensure_dir_exists(os.path.join(dataset_output_dir, "categories"))
            
            print(f"正在处理 {dataset_name} 数据集，输出到: {dataset_output_dir}")
            
            # 处理数据集
            process_deepfashion2(
                image_dir,
                annos_dir,
                dataset_output_dir,
                max_images=max_images
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])