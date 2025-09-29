#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_builder.py

基于 data/logs/metadata.csv 做 stratified split 并输出 metadata_{split}.csv 或构建 data/processed/{train,val,test}/<label>/ 图片软链接。
"""

from pathlib import Path
import argparse
import shutil
import os
import sys
import math
import warnings
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------- Defaults --------------------
PROJECT_ROOT = Path.cwd()
DEFAULT_METADATA = PROJECT_ROOT / "data" / "logs" / "metadata.csv"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"

# -------------------- Helper --------------------
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

def build_label_column(df: pd.DataFrame, stratify_by: str) -> Tuple[pd.DataFrame, str]:
    if stratify_by in ("plant", "plant_only"):
        if "plant" not in df.columns:
            raise ValueError("metadata 缺少 'plant' 字段")
        return df, "plant"
    if stratify_by in ("plant+disease", "plant_disease", "plant_disease_combo"):
        if "disease" not in df.columns:
            warnings.warn("metadata 缺少 'disease' 字段，降级为按 plant 分层")
            return df, "plant"
        df = df.copy()
        df["plant__disease"] = df["plant"].astype(str) + "__" + df["disease"].astype(str)
        return df, "plant__disease"
    raise ValueError("unsupported stratify_by: " + str(stratify_by))

def stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float, test_ratio: float, seed: int = 42):
    total = len(df)
    if total == 0:
        raise ValueError("empty dataframe")
    if val_ratio + test_ratio <= 0:
        df["split"] = "train"
        return df

    holdout_ratio = val_ratio + test_ratio
    train_ratio = 1.0 - holdout_ratio
    if train_ratio <= 0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()
    if min_count < 2:
        warnings.warn(f"类别样本过少（最小类样本数 {min_count}），stratified split 可能失败")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    y = df[label_col].values
    try:
        train_idx, rest_idx = next(sss1.split(df, y))
    except Exception as e:
        warnings.warn(f"首次 stratified split 失败（{e}），改用随机 split")
        perm = df.sample(frac=1.0, random_state=seed).index
        train_n = int(math.floor(len(df) * train_ratio))
        train_idx = perm[:train_n]
        rest_idx = perm[train_n:]

    train_df = df.iloc[train_idx].copy()
    rest_df = df.iloc[rest_idx].copy()

    if val_ratio == 0:
        val_df = rest_df.iloc[[]].copy()
        test_df = rest_df.copy()
    elif test_ratio == 0:
        val_df = rest_df.copy()
        test_df = rest_df.iloc[[]].copy()
    else:
        val_within_rest = val_ratio / (val_ratio + test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_within_rest, random_state=seed+1)
        y_rest = rest_df[label_col].values
        try:
            val_idx_sub, test_idx_sub = next(sss2.split(rest_df, y_rest))
        except Exception as e:
            warnings.warn(f"第二次 stratified split 失败（{e}），改用随机 split")
            perm = rest_df.sample(frac=1.0, random_state=seed+1).index
            val_n = int(math.floor(len(rest_df) * val_within_rest))
            val_idx_sub = perm[:val_n]
            test_idx_sub = perm[val_n:]
        val_df = rest_df.iloc[val_idx_sub].copy()
        test_df = rest_df.iloc[test_idx_sub].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out_df = pd.concat([train_df, val_df, test_df], axis=0)
    return out_df.reset_index(drop=True)

def write_split_csvs(df: pd.DataFrame, output_dir: Path):
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        csv_path = output_dir / f"metadata_{split}.csv"
        backup_file(csv_path)
        split_df.to_csv(csv_path, index=False)
        print(f"✅ 写入 {csv_path} ({len(split_df)} rows)")

def build_processed_dirs(df: pd.DataFrame, output_dir: Path, use_copy=False):
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
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
            except Exception as e:
                warnings.warn(f"无法创建 {dst} -> {src} ({e})，尝试 copy")
                shutil.copy2(src, dst)
    print(f"🎯 processed 目录构建完成：{output_dir}")

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--stratify-by", type=str, default="plant")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out-processed", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"❌ metadata CSV 不存在: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    df, label_col = build_label_column(df, args.stratify_by)
    df_split = stratified_split(df, label_col, args.val_ratio, args.test_ratio)

    if args.dry_run:
        print("⚠️ dry-run 模式，未写文件")
        print(df_split["split"].value_counts())
        return

    out_dir = DEFAULT_PROCESSED
    safe_mkdir(out_dir)
    write_split_csvs(df_split, out_dir)

    if args.out_processed:
        build_processed_dirs(df_split, out_dir, use_copy=args.copy)

if __name__ == "__main__":
    main()

