"""
图像预处理：resize、normalize、增强
- 可复用为 pipeline 被 dataset_builder 调用
"""
from pathlib import Path
from PIL import Image
import os
import shutil

def resize_image(input_path: Path, output_path: Path, size=(256,256)):
    os.makedirs(output_path.parent, exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    img.save(output_path, quality=95)

def preprocess_dataset(src_dir: Path, dst_dir: Path, size=(256,256)):
    """
    遍历 src_dir/<label>/*.jpg，统一 resize 并拷贝到 dst_dir/<label>/
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for label_dir in src_dir.iterdir():
        if not label_dir.is_dir():
            continue
        out_label_dir = dst_dir / label_dir.name
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for img_file in label_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            out_file = out_label_dir / img_file.name
            resize_image(img_file, out_file, size=size)
    print(f"Preprocess done. Output to {dst_dir}")
