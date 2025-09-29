# 在项目根运行：
# python src/data_collection/all_in_one_build_full.py --download-pv
# 就会自动执行扫描、去重、生成 metadata、构建 processed。

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all_in_one_build_full.py
一体化构建 Plant/PlantDisease 数据集
- 扫描本地数据
- 下载 PlantVillage（可选）
- 去重 (md5 + phash)
- 构建 stratified train/val/test
- 生成 data/processed/{train,val,test}/<label>/ 的软链接或复制
- 生成 metadata CSV + JSON
"""

import os
from pathlib import Path
import argparse
import shutil
import warnings
from tqdm import tqdm
import pandas as pd
from PIL import Image
import hashlib
import imagehash

# 导入 dataset_builder 核心函数
from dataset_builder import stratified_split, build_label_column, safe_mkdir, backup_file

PROJECT_ROOT = Path.cwd()
RAW_LOCAL = PROJECT_ROOT / "data/raw/local_dataset"
RAW_EXTERNAL = PROJECT_ROOT / "data/raw/external_dataset"
LOGS_DIR = PROJECT_ROOT / "data/logs"
METADATA_CSV = LOGS_DIR / "metadata.csv"
METADATA_JSON = LOGS_DIR / "metadata.json"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

# -------------------- 工具函数 --------------------
def download_plantvillage(target_dir: Path):
    safe_mkdir(target_dir)
    if any(target_dir.iterdir()):
        print(f"PlantVillage 目录已有内容，跳过下载: {target_dir}")
        return
    print("⚡ 请手动下载 PlantVillage 数据集并解压到", target_dir)

def scan_folder_to_metadata(root: Path, source_name: str, use_phash=True):
    rows = []
    if not root.exists():
        print(f"目录不存在: {root}")
        return rows
    for plant_dir in sorted(root.iterdir()):
        if not plant_dir.is_dir():
            continue
        plant = plant_dir.name
        for file in tqdm(sorted(plant_dir.rglob("*")), desc=f"Scanning {plant_dir}"):
            if file.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.tiff','.webp'):
                continue
            try:
                with Image.open(file) as img:
                    w,h = img.size
                    mode = img.mode
                    phash_val = str(imagehash.phash(img)) if use_phash else None
            except:
                w=h=None
                mode=""
                phash_val = None
            md5 = hashlib.md5(file.read_bytes()).hexdigest()
            rows.append({
                "filepath": str(file.relative_to(PROJECT_ROOT)),
                "plant": plant,
                "filename": file.name,
                "width": w,
                "height": h,
                "mode": mode,
                "filesize": file.stat().st_size,
                "md5": md5,
                "phash": phash_val,
                "source": source_name
            })
    return rows

def deduplicate(rows, use_phash=True):
    seen_md5 = set()
    seen_phash = set()
    unique=[]
    for r in rows:
        if r['md5'] in seen_md5:
            continue
        if use_phash and r['phash'] in seen_phash:
            continue
        seen_md5.add(r['md5'])
        if use_phash:
            seen_phash.add(r['phash'])
        unique.append(r)
    return unique

def write_metadata(rows, csv_path: Path, json_path: Path):
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 备份旧文件
    backup_file(csv_path)
    backup_file(json_path)
    df.to_csv(csv_path,index=False)
    df.to_json(json_path,orient="records",force_ascii=False,indent=2)
    print(f"✅ metadata 写入 {csv_path} ({len(df)} 行) + {json_path}")
    return df

def build_processed(df, val_ratio, test_ratio, copy=False):
    df,label_col = build_label_column(df,"plant")
    df = stratified_split(df,label_col,val_ratio,test_ratio)
    for split in ["train","val","test"]:
        sub = df[df['split']==split]
        for idx,row in tqdm(sub.iterrows(), total=len(sub), desc=f"Building {split}"):
            src = PROJECT_ROOT / row['filepath']
            dst = PROCESSED_DIR / split / row[label_col] / src.name
            safe_mkdir(dst.parent)
            if copy:
                shutil.copy2(src,dst)
            else:
                try:
                    if dst.exists():
                        dst.unlink()
                    os.symlink(src.resolve(), dst)
                except Exception:
                    shutil.copy2(src,dst)
    print(f"✅ processed 构建完成，路径: {PROCESSED_DIR}")

# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-pv",action="store_true",help="下载 PlantVillage 数据集")
    parser.add_argument("--copy",action="store_true",help="不使用软链接，直接复制文件到 processed")
    parser.add_argument("--val-ratio",type=float,default=0.15)
    parser.add_argument("--test-ratio",type=float,default=0.1)
    parser.add_argument("--no-phash",action="store_true",help="禁用 phash 去重")
    args = parser.parse_args()

    if args.download_pv:
        download_plantvillage(RAW_EXTERNAL)

    # 扫描 local + external
    rows_local = scan_folder_to_metadata(RAW_LOCAL,"local",use_phash=not args.no_phash)
    rows_ext   = scan_folder_to_metadata(RAW_EXTERNAL,"external",use_phash=not args.no_phash)
    all_rows = deduplicate(rows_local + rows_ext,use_phash=not args.no_phash)

    df = write_metadata(all_rows, METADATA_CSV, METADATA_JSON)

    # 构建 processed train/val/test
    build_processed(df,args.val_ratio,args.test_ratio,copy=args.copy)
    print("🎉 全流程完成")

if __name__=="__main__":
    main()
