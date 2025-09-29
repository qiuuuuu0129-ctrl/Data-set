# scripts/gen_dummy_data.py
"""
生成用于测试的假植物图片集：data/raw/local_dataset/<plant_name>/imgXXX.jpg
- 每个类别生成若干张简单合成图片（带文字的彩色背景）
- 可配置类别数与每类图片数
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import os

ROOT = Path.cwd()
OUT_BASE = ROOT / "data" / "raw" / "local_dataset"
PLANTS = ["tomato", "potato", "wheat"]
IMAGES_PER_PLANT = 30
IMG_SIZE = (256, 256)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def create_image(text: str, out_path: Path, size=IMG_SIZE):
    # simple colored bg with text
    img = Image.new("RGB", size, (random.randint(100,255), random.randint(100,255), random.randint(100,255)))
    draw = ImageDraw.Draw(img)
    try:
        # PIL default font
        f = ImageFont.load_default()
    except Exception:
        f = None
    tw, th = draw.textsize(text, font=f)
    draw.text(((size[0]-tw)/2, (size[1]-th)/2), text, fill=(0,0,0), font=f)
    img.save(out_path, quality=90)

def main():
    print("生成假数据到：", OUT_BASE)
    for plant in PLANTS:
        out_dir = OUT_BASE / plant
        ensure_dir(out_dir)
        for i in range(IMAGES_PER_PLANT):
            name = f"{plant}_{i:03d}.jpg"
            path = out_dir / name
            # vary text a bit to create slightly different images
            text = f"{plant} #{i}"
            create_image(text, path)
    print("完成。每类生成", IMAGES_PER_PLANT, "张，总类:", len(PLANTS))

if __name__ == "__main__":
    main()
