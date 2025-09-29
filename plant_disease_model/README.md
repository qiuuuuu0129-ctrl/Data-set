#这是一个为plantai项目大模型及深度学习算法而设立的可读性文件
#当然可以，我帮你把你目前所有主要文件的用途、运行方法和调用顺序整理成一个清晰的合集。这样你就能一目了然整个采集到处理流程。


## **1️⃣ capture_dataset.py — 专业采集脚本**

**位置建议**：项目根或 `src/data_collection/`

**功能**：

* 使用摄像头采集植物图片
* 支持图像质量检测（模糊、亮度）
* 可选人脸检测跳过
* 会话内 phash 去重（避免重复保存重复帧）
* 保存 metadata CSV + JSON 日志
* 可选后台上传
* 支持交互式输入标签或批量 labels

**运行方法**：

```bash
python capture_dataset_full.py
```

**交互示例**：

* 批量采集：输入 `y`，然后输入英文逗号分隔的类别列表
* 单次采集：逐个输入类别名称，输入 `exit` 停止

---

## **2️⃣ gather_local.py — 本地图片扫描 & metadata 生成**

**位置建议**：`src/data_collection/`

**功能**：

* 扫描 `data/raw/local_dataset/<plant>` 文件夹
* 支持 jpg/png/jpeg/webp 等格式
* 计算 md5、phash（可选依赖 imagehash）
* 支持去重（基于 md5 或 phash 汉明距离阈值）
* 输出 CSV 到 `data/logs/metadata.csv`
* 可选 `--dedup-action move` 将重复图片移动到 `data/logs/duplicates/`
* 支持 `--dry-run` 仅打印不写文件

**运行方法**：

```bash
python src/data_collection/gather_local.py
# 示例：移动重复图片，phash 阈值 5
python src/data_collection/gather_local.py --dedup-action move --phash-threshold 5
```

---

## **3️⃣ dataset_builder.py — 数据集划分 & 构建 processed**

**位置建议**：`src/data_collection/`

**功能**：

* 基于 `data/logs/metadata.csv` 做 train/val/test stratified split
* 支持 `--stratify-by plant` 或 `plant+disease`
* 可选创建 `data/processed/{train,val,test}/<label>/` 并生成软链接或复制图片
* 支持 dry-run 不写文件

**运行方法**：

```bash
# 简单划分
python src/data_collection/dataset_builder.py

# 指定划分比例，生成 processed，复制文件
python src/data_collection/dataset_builder.py --val-ratio 0.15 --test-ratio 0.10 --out-processed --copy

# dry-run
python src/data_collection/dataset_builder.py --dry-run
```

---

## **4️⃣ all_in_one_build_full.py（可选） — 升级整合版**

**位置建议**：`src/data_collection/`

**功能**：

* 自动整合 gather_local + dataset_builder
* 支持可选下载、扫描、去重、划分、构建 processed
* 可以一条命令完成整个数据集整理流程

**运行方法**：

```bash
python src/data_collection/all_in_one_build_full.py --download-pv
```

参数可选：

* `--dry-run`
* `--copy`（强制复制而非软链接）
* `--stratify-by plant+disease`

---

## **5️⃣ 使用顺序建议**

1. **采集新图片**

   ```bash
   python capture_dataset_full.py
   ```

   * 或者已有本地图片，可跳过此步

2. **扫描本地图片生成 metadata**

   ```bash
   python src/data_collection/gather_local.py --dedup-action move
   ```

3. **划分数据集 & 构建 processed**

   ```bash
   python src/data_collection/dataset_builder.py --val-ratio 0.15 --test-ratio 0.10 --out-processed
   ```

4. **升级整合版（可替代 2+3 步）**

   ```bash
   python src/data_collection/all_in_one_build_full.py --download-pv
   ```

---

从采集→扫描→去重→划分→生成 processed

---

# **README.md**（完整项目版）

```markdown
# Plant Disease Model

## 项目简介
本项目是一个面向植物图像识别与病虫害分析的完整系统模板，包含从数据收集、处理、训练、推理到 Web 展示的全流程。  
设计目标是提供初始可运行模板，便于二次开发和快速实验。

---

## 项目结构

```

plant_disease_model/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/
│   ├── pytorch/
│   │   └── save_load_demo.py       # 演示 PyTorch 模型保存与加载
│   └── tflite/                     # 可存放 TFLite 模型权重
├── data/
│   ├── logs/
│   │   └── json/
│   │       └── write_demo.py       # JSON 日志写入示例
│   ├── processed/                  # 数据处理后的训练/验证/测试集（可自动生成）
│   └── raw/
│       └── local_dataset/
│           └── generate_demo_structure.py  # 占位文件夹/目录生成脚本
├── configs/
│   └── dataset_config.yaml         # 数据划分与处理配置
├── src/
│   ├── data_collection/
│   │   └── dataset_builder.py     # 根据 metadata 构建训练/验证/测试集
│   ├── datasets/                   # 数据集类定义与 transform
│   ├── models/
│   │   └── plant_classifier.py     # 示例分类模型
│   ├── train/                      # 训练脚本存放目录
│   ├── infer/                      # 推理脚本存放目录
│   ├── utils/                      # 工具函数存放目录
│   └── web/                        # Web 端接口代码
├── web/
│   └── app.py                      # Flask Web 示例
└── scripts/
└── gen_dummy_data.py           # 示例脚本，原占位用途，可替换为真实工具

````

---

## 目录说明

### 1. checkpoints/
- `pytorch/`：PyTorch 模型权重存放目录
  - `save_load_demo.py`：示例如何保存/加载 PyTorch 模型
- `tflite/`：可存放 TFLite 模型权重，方便移动端部署

### 2. data/
- `logs/json/`：实验日志或训练信息
  - `write_demo.py`：演示写入 JSON 日志
- `processed/`：数据处理后的训练/验证/测试集
- `raw/local_dataset/`：本地原始数据集
  - `generate_demo_structure.py`：生成占位目录及示例图片文件夹

### 3. configs/
- `dataset_config.yaml`：用于数据划分、transform 配置及训练参数

### 4. src/
- `data_collection/dataset_builder.py`：根据 metadata.csv 构建 stratified split 数据集
- `datasets/`：数据集类及 transform
- `models/plant_classifier.py`：示例 CNN 模型，可扩展
- `train/`：训练脚本存放目录，可按项目需求扩展
- `infer/`：推理脚本存放目录
- `utils/`：工具函数，如图像处理、日志管理等
- `web/`：Web 端接口和调用模型的逻辑

### 5. web/app.py
- Flask 示例，可直接运行 Web 服务
- 提供图像上传接口并返回预测结果

### 6. scripts/
- `gen_dummy_data.py`：占位脚本，可替换为实际数据采集/生成工具

---

## 安装依赖

建议 Python >=3.10，创建虚拟环境并安装依赖：

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
````

`requirements.txt` 示例：

```
pandas
scikit-learn
torch
flask
Pillow
opencv-python
imagehash
PyYAML
```

---

## 快速运行示例

### 1. 生成占位数据集目录

```bash
python data/raw/local_dataset/generate_demo_structure.py
```

### 2. 写入 JSON 日志示例

```bash
python data/logs/json/write_demo.py
```

### 3. PyTorch 模型保存/加载示例

```bash
python checkpoints/pytorch/save_load_demo.py
```

### 4. 构建训练/验证/测试集

```bash
python src/data_collection/dataset_builder.py --metadata data/logs/metadata.csv --val-ratio 0.15 --test-ratio 0.1 --out-processed
```

### 5. 启动 Web 服务

```bash
python web/app.py
# 访问 http://127.0.0.1:5000
```

---

## 使用指南

1. 将 `raw/local_dataset/` 替换为真实植物图像数据集
2. 修改 `dataset_config.yaml` 配置数据处理参数
3. 在 `models/` 中实现复杂模型结构
4. 在 `train/` 编写训练脚本，`infer/` 编写推理脚本
5. 在 `web/` 中实现前端或接口服务

---

## 项目拓展

* 支持 TFLite 模型部署到移动端
* 增加数据增强模块
* 添加多标签识别（如病虫害种类 + 严重程度）
* 集成实时摄像头推理

```

```
