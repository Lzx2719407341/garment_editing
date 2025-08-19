# garment_editing

df2_preprocess.py 是专门用来进行DeepFashion2数据集的数据预处理代码，可在data_config.yaml中设置需要处理的最大图片数量max_images
其运行命令如下：
```bash
python -u "你的路径\garment_editing\data\df2_preprocess.py" "你的路径\garment_editing\configs\data_config.yaml"
```
运行结果的文件框架如下：
```bash

./garment_editing/data/processed/deepfashion2/  # 基础输出目录（来自配置的output_dir）
├── train/  # 训练集处理结果（来自配置的train_output_dir）
│   ├── garment_masks/  # 服装掩码图像
│   │   ├── {json文件名}_{item_key}.png  # 例：000001_1.png（对应JSON文件000001.json的item_key=1的掩码）
│   │   └── ...……
│   ├── edge_masks/  # 服装边缘掩码图像（宽度5像素）
│   │   ├── {json文件名}_{item_key}.png
│   │   └── ...…│└......│└......
│   ├── prompts/  # 服装属性文本描述
│   │   ├── {json文件名}_{item_key}.txt  # 内容例："red dress style casual slightly occluded"
│   │   └── ...……
│   ├── images/  # 原始图像（按标注项拆分保存）
│   │   ├── {json文件名}_{item_key}.jpg
│   │   └── ...…│└......
│   ├── bounding_boxes/  # 边界框坐标（x, y, width, height）
│   │   ├── {json文件名}_{item_key}.txt  # 内容例："10 20 100 200"
│   │   └── ...…│└......
│   ├── landmarks/  # 关键点信息（JSON格式）
│   │   ├── {json文件名}_{item_key}.json
│   │   └── ...…│└......
│   └── categories/  # 服装类别信息
│       ├── {json文件名}_{item_key}.txt  # 内容例："dress"
│       └── ...│美国…
│
├── validation/  # 验证集处理结果（来自配置的validation_output_dir）
│   ├── garment_masks/《我的面具》
│   ├── edge_masks/（英文）
│   ├── prompts/我的意思是：
│   ├── images/我的意象/
│   ├── bounding_boxes/
│   ├── landmarks/
│   └── categories/
│   # 子目录内容结构同train
│
└── test/  # 测试集处理结果（来自配置的test_output_dir）
    ├── garment_masks/
    ├── edge_masks/
    ├── prompts/   ├──敏于/
    ├── images/   ├──/图像
    ├── bounding_boxes/
    ├── landmarks/   ├──地标/
    └── categories/   └──类别/
    # 子目录内容结构同train（注：测试集无标注，部分目录为空）
```

hp_preprocess.py 是专门用来进行Human_Parsing_Dataset文件夹下CIHP数据集的数据预处理代码，可在data_config.yaml中设置需要处理的最大图片数量max_images
其运行命令如下：
```bash
python -u "你的路径\garment_editing\data\hp_preprocess.py" "你的路径\garment_editing\configs\data_config.yaml"
```
运行结果的文件框架如下：
```bash
./garment_editing/data/processed/human_parsing/CIHP  # 基础输出目录（来自配置的output_dir）
├── train/                 # 训练集输出目录
│   ├── images/            # 原始图像（.jpg格式）
│   │   ├── {img_id1}.jpg
│   │   ├── {img_id2}.jpg
│   │   └── ...…
│   ├── human_masks/       # 人体非服装区域掩码（.png格式，255表示保留区域）
│   │   ├── {img_id1}.png
│   │   ├── {img_id2}.png
│   │   └── ...…
│   └── edge_masks/        # 人体非服装区域边缘掩码（.png格式，边缘为255）
│       ├── {img_id1}.png
│       ├── {img_id2}.png
│       └── ...│美国…
│
├── validation/            # 验证集输出目录（结构同训练集）
│   ├── images/我的意象/
│   ├── human_masks/
│   └── edge_masks/
│
└── test/                  # 测试集输出目录（结构根据是否有标注略有差异）
    ├── images/            # 必存在，存放测试集原始图像
    │   ├── {img_id1}.jpg
    │   └── ...
    ├── human_masks/       # 仅当测试集有标注目录时存在（生成默认为空）
    │   ├── {img_id1}.png
    │   └── ...
    └── edge_masks/        # 仅当测试集有标注目录时存在（生成默认为空）
        ├── {img_id1}.png
        └── ...
```
