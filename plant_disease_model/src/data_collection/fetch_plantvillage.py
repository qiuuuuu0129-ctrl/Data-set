#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_build_dataset.py

一条命令从本地图片或已有 dataset 自动生成 ready-to-train 数据集。
包含：
1. 收集本地图片
2. 整理成统一目录结构
3. 分层划分 train/val/test
4. 输出 metadata CSV 和 processed 软链接/复制
"""

from pathlib import Path
import argparse
import shutil
import os
import sys
import warnings
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_plantvillage.py

自动下载 PlantVillage 开源数据集并整理到 data/raw/external_dataset
"""

import requests, zipfile, io
from pathlib import Path
import shutil

PROJECT_ROOT = Path.cwd()
EXTERNAL_DATA = PROJECT_ROOT / "data" / "raw" / "external_dataset"
safe_mkdir = lambda p: p.mkdir(parents=True, exist_ok=True)

def download_and_extract(url: str, out_dir: Path):
    print(f"🔽 下载 {url} ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise ValueError(f"下载失败: {url}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        print(f"📦 解压到 {out_dir} ...")
        z.extractall(out_dir)
    print("✅ 下载并解压完成")

def main():
    safe_mkdir(EXTERNAL_DATA)

    # 官方 PlantVillage dataset URL（RGB images）
    plantvillage_url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/3c9e9caa-3f2a-4b34-a74b-c9d7b2a81f5b/file_downloaded"
    download_and_extract(plantvillage_url, EXTERNAL_DATA)

    # 可选：对文件夹进行归一化，例如把病害名放到 parent__disease
    for subdir in EXTERNAL_DATA.iterdir():
        if subdir.is_dir():
            # 假设 folder 名格式: plant_disease
            if "_" in subdir.name:
                plant, disease = subdir.name.split("_", 1)
                target = EXTERNAL_DATA / f"{plant}__{disease}"
                if target != subdir:
                    shutil.move(subdir, target)
    print("🎯 external_dataset 整理完成")

if __name__ == "__main__":
    main()

# -------------------- Defaults --------------------
PROJECT_ROOT = Path.cwd()
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "local_dataset"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_CSV = LOG_DIR / "metadata.csv"

# -------------------- Helpers --------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def backup_file(p: Path):
    if not p.exists():
        return None
    idx = 0
    while True:
        backup = p.with_suffix(p.suffix + f".bak{idx}")
        if not backup.exists():
            p.rename(backup)
            return backup
        idx += 1

def build_label_column(df: pd.DataFrame, stratify_by: str):
    if stratify_by in ("plant", "plant_only"):
        if "plant" not in df.columns:
            raise ValueError("metadata 缺少 'plant' 字段")
        return df, "plant"
    if stratify_by in ("plant+disease", "plant_disease"):
        if "disease" not in df.columns:
            warnings.warn("metadata 缺少 'disease' 字段，降级为按 plant 分层")
            return df, "plant"
        df = df.copy()
        df["plant__disease"] = df["plant"].astype(str) + "__" + df["disease"].astype(str)
        return df, "plant__disease"
    raise ValueError(f"unsupported stratify_by: {stratify_by}")

def stratified_split(df, label_col, val_ratio, test_ratio, seed=42):
    total = len(df)
    if total == 0:
        raise ValueError("empty dataframe")
    if val_ratio + test_ratio <= 0:
        df["split"] = "train"
        return df

    holdout_ratio = val_ratio + test_ratio
    train_ratio = 1.0 - holdout_ratio
    class_counts = df[label_col].value_counts()
    if class_counts.min() < 2:
        warnings.warn(f"最小类样本数 {class_counts.min()}, stratified split 可能失败")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    y = df[label_col].values
    try:
        train_idx, rest_idx = next(sss1.split(df, y))
    except Exception:
        perm = df.sample(frac=1.0, random_state=seed).index
        train_n = int(len(df) * train_ratio)
        train_idx = perm[:train_n]
        rest_idx = perm[train_n:]

    train_df = df.iloc[train_idx].copy()
    rest_df = df.iloc[rest_idx].copy()

    if val_ratio == 0:
        val_df = rest_df.iloc[[]].copy()
        test_df = rest_df
    elif test_ratio == 0:
        val_df = rest_df
        test_df = rest_df.iloc[[]].copy()
    else:
        val_within_rest = val_ratio / (val_ratio + test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_within_rest, random_state=seed+1)
        y_rest = rest_df[label_col].values
        try:
            val_idx_sub, test_idx_sub = next(sss2.split(rest_df, y_rest))
        except Exception:
            perm = rest_df.sample(frac=1.0, random_state=seed+1).index
            val_n = int(len(rest_df) * val_within_rest)
            val_idx_sub = perm[:val_n]
            test_idx_sub = perm[val_n:]
        val_df = rest_df.iloc[val_idx_sub].copy()
        test_df = rest_df.iloc[test_idx_sub].copy()

    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"
    return pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

def write_split_csvs(df, output_dir):
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if split_df.empty: continue
        path = output_dir / f"metadata_{split}.csv"
        backup_file(path)
        split_df.to_csv(path, index=False)
        print(f"✅ 写入 {path} ({len(split_df)} rows)")

def build_processed_dirs(df, output_dir, use_copy=False):
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        for _, row in split_df.iterrows():
            src = Path(row["filepath"])
            label = str(row.get("plant__disease") or row.get("plant"))
            dst_dir = output_dir / split / label
            safe_mkdir(dst_dir)
            dst = dst_dir / src.name
            try:
                if use_copy or os.name == "nt":
                    shutil.copy2(src, dst)
                else:
                    if dst.exists(): dst.unlink()
                    dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)
    print(f"🎯 processed 目录构建完成：{output_dir}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(RAW_DATA), help="本地原始图片根目录")
    parser.add_argument("--stratify-by", type=str, default="plant")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out-processed", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    safe_mkdir(LOG_DIR)
    safe_mkdir(PROCESSED_DIR)
    safe_mkdir(RAW_DATA)

    # -------------------- Step1: Gather metadata --------------------
    # 假设每个子目录为 plant 或 plant__disease
    records = []
    for file_path in RAW_DATA.rglob("*.*"):
        if file_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        parent = file_path.parent.name
        # 支持 plant 或 plant__disease
        if "__" in parent:
            plant, disease = parent.split("__", 1)
        else:
            plant, disease = parent, ""
        records.append({"filepath": str(file_path), "plant": plant, "disease": disease})
    df_meta = pd.DataFrame(records)
    df_meta.to_csv(METADATA_CSV, index=False)
    print(f"✅ 生成 metadata: {METADATA_CSV} ({len(df_meta)} rows)")

    # -------------------- Step2: Stratified split --------------------
    df_meta, label_col = build_label_column(df_meta, args.stratify_by)
    df_split = stratified_split(df_meta, label_col, args.val_ratio, args.test_ratio)

    if args.dry_run:
        print("⚠️ dry-run 模式")
        print(df_split["split"].value_counts())
        return

    write_split_csvs(df_split, PROCESSED_DIR)

    # -------------------- Step3: 构建 processed 目录 --------------------
    if args.out_processed:
        build_processed_dirs(df_split, PROCESSED_DIR, use_copy=args.copy)

if __name__ == "__main__":
    main()
