#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#依赖：Pillow, pandas，可选 imagehash（用于 pHash 去重，推荐安装 pip install imagehash）。
#默认行为不会移动或删除图片，只会创建/合并 data/logs/metadata.csv，避免误操作。
# --dedup-action move 或 --move-duplicates 可以把检测到的重复文件移动到 data/logs/duplicates/（默认不会启用）。
# 如果数据量很大（>10k 图）且启用了 phash 去重，phash 检测是 O(n^2)，可以后续改成 LSH /近邻索引。

"""
gather_local.py
扫描 data/raw/local_dataset 下各植物子文件夹，生成 data/logs/metadata.csv
功能：
 - 遍历每种植物的图片文件（支持 jpg/png/jpeg/webp 等）
 - 计算文件 md5、pHash（可选依赖 imagehash + PIL）
 - 记录 filepath, plant, filename, width, height, filesize, md5, phash, mode, source
 - 支持去重检测（基于 md5 或 phash 汉明距离阈值）
 - 支持 --dedup-action {ignore, move}（默认 ignore；move 会把重复图片移动到 data/logs/duplicates/）
 - 支持 --dry-run（只打印，不改文件/不写 CSV）
 - 输出 CSV 到 data/logs/metadata.csv（默认会合并已有 metadata 并去重）
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Optional dependency; if not available, phash will be skipped
try:
    import imagehash  # pip install imagehash
    HAS_IMAGEHASH = True
except Exception:
    HAS_IMAGEHASH = False

import pandas as pd

# ---------- Configurable constants ----------
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
PROJECT_ROOT = Path.cwd()  # assume run from project root; adjust if needed
RAW_LOCAL = PROJECT_ROOT / "data" / "raw" / "local_dataset"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
METADATA_CSV = LOGS_DIR / "metadata.csv"
DUP_DIR = LOGS_DIR / "duplicates"

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gather_local")


# ---------- Utility functions ----------
def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path) -> Optional[str]:
    if not HAS_IMAGEHASH:
        return None
    try:
        with Image.open(path) as img:
            ph = imagehash.phash(img)
            return str(ph)
    except Exception:
        return None


def safe_open_image(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            return img.width, img.height, img.mode
    except UnidentifiedImageError:
        return None, None, None
    except Exception:
        return None, None, None


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS


# ---------- Main scanner ----------
def scan_local_folder(root: Path) -> List[Dict]:
    """
    Walks local_dataset/<plant_name> structure and returns a list of row dicts
    """
    rows = []
    if not root.exists():
        logger.warning(f"{root} 不存在。请先按 README 把图片放到 data/raw/local_dataset/<plant>/")
        return rows

    for plant_dir in sorted(root.iterdir()):
        if not plant_dir.is_dir():
            continue
        plant = plant_dir.name
        logger.info(f"扫描植物: {plant}")
        for file in sorted(plant_dir.rglob("*")):
            if not is_image_file(file):
                continue
            rel = file.relative_to(PROJECT_ROOT).as_posix()
            width, height, mode = safe_open_image(file)
            md5 = compute_md5(file)
            phash = compute_phash(file) if HAS_IMAGEHASH else None
            filesize = file.stat().st_size
            rows.append({
                "filepath": rel,
                "plant": plant,
                "filename": file.name,
                "width": width if width is not None else "",
                "height": height if height is not None else "",
                "mode": mode if mode is not None else "",
                "filesize": filesize,
                "md5": md5,
                "phash": phash if phash is not None else "",
                "source": "local",
                "notes": ""
            })
    return rows


# ---------- Dedup logic ----------
def find_duplicates(rows: List[Dict], existing_md5s: set, phash_distance_threshold: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (unique_rows, duplicate_rows)
    - md5 exact match => duplicate
    - else if phash available: compare to each other and to existing_phashes
      and mark duplicate if hamming distance <= phash_distance_threshold
    Note: O(n^2) on phash checks; acceptable for moderate datasets (thousands). For large sets use LSH.
    """
    unique = []
    duplicates = []

    # build md5 and phash sets for existing metadata if provided
    seen_md5 = set(existing_md5s)
    seen_phashes: List[str] = []

    for r in rows:
        md5 = r.get("md5")
        ph = r.get("phash")
        if md5 and md5 in seen_md5:
            duplicates.append({**r, "dup_reason": "md5_existing"})
            continue

        is_dup = False
        if md5 and md5 in seen_md5:
            is_dup = True

        # Check phash against already seen phashes for near-duplicates
        if not is_dup and ph:
            for existing_ph in seen_phashes:
                # convert to imagehash.ImageHash by parsing hex
                try:
                    # Compare by hamming distance on hex strings using imagehash lib if available
                    if HAS_IMAGEHASH:
                        h1 = imagehash.hex_to_hash(ph)
                        h2 = imagehash.hex_to_hash(existing_ph)
                        dist = (h1 - h2)
                        if dist <= phash_distance_threshold:
                            duplicates.append({**r, "dup_reason": f"phash_near_{dist}"})
                            is_dup = True
                            break
                    else:
                        break
                except Exception:
                    continue

        if is_dup:
            continue

        # unique: add to seen lists
        unique.append(r)
        if md5:
            seen_md5.add(md5)
        if ph:
            seen_phashes.append(ph)

    return unique, duplicates


# ---------- CSV helpers ----------
def read_existing_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.warning(f"读取已有 metadata 失败: {e}")
        return pd.DataFrame()


def write_metadata(df: pd.DataFrame, path: Path, dry_run: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logger.info(f"[dry-run] 不写入 CSV，行数: {len(df)}")
        return
    df.to_csv(path, index=False)
    logger.info(f"已写入 metadata -> {path} （{len(df)} 行）")


# ---------- Main entry ----------
def main(args):
    logger.info("开始扫描 local_dataset ...")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.move_duplicates:
        DUP_DIR.mkdir(parents=True, exist_ok=True)

    existing_df = read_existing_metadata(METADATA_CSV)
    existing_md5s = set(existing_df.get("md5", pd.Series(dtype=str)).dropna().astype(str).tolist())

    scanned = scan_local_folder(RAW_LOCAL)
    logger.info(f"扫描完成，共发现 {len(scanned)} 张图片（候选）")

    # Dedup
    unique_rows, duplicate_rows = find_duplicates(scanned, existing_md5s, phash_distance_threshold=args.phash_threshold)
    logger.info(f"去重后：保留 {len(unique_rows)}，检测到 {len(duplicate_rows)} 重复")

    # If dedup-action == move, move duplicates to logs/duplicates
    if args.dedup_action == "move" and not args.dry_run:
        for d in duplicate_rows:
            src = PROJECT_ROOT / d["filepath"]
            if not src.exists():
                continue
            dst = DUP_DIR / src.name
            # avoid overwrite
            i = 0
            while dst.exists():
                i += 1
                dst = DUP_DIR / f"{src.stem}_{i}{src.suffix}"
            try:
                src.rename(dst)
                logger.info(f"移动重复文件: {src} -> {dst}")
                # update filepath in duplicate row for logging
                d["filepath_old"] = str(src.relative_to(PROJECT_ROOT).as_posix())
                d["filepath"] = str(dst.relative_to(PROJECT_ROOT).as_posix())
            except Exception as e:
                logger.warning(f"移动失败 {src}: {e}")

    # Merge with existing metadata (append new unique rows)
    combined_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()
    new_df = pd.DataFrame(unique_rows)
    if not new_df.empty:
        # Ensure columns order / fill missing
        columns = ["filepath", "plant", "filename", "width", "height", "mode", "filesize", "md5", "phash", "source", "notes"]
        for c in columns:
            if c not in new_df.columns:
                new_df[c] = ""
        new_df = new_df[columns]
        combined_df = pd.concat([combined_df, new_df], ignore_index=True, sort=False)
    else:
        logger.info("无新图片追加到 metadata。")

    # Save metadata
    write_metadata(combined_df, METADATA_CSV, dry_run=args.dry_run)

    # Save duplicate log JSON/CSV for audit
    if duplicate_rows:
        dup_df = pd.DataFrame(duplicate_rows)
        dup_path = LOGS_DIR / "duplicates.csv"
        if not args.dry_run:
            dup_df.to_csv(dup_path, index=False)
            logger.info(f"重复项记录已保存 -> {dup_path}")
        else:
            logger.info(f"[dry-run] 重复项记录（{len(duplicate_rows)} 行）未写入磁盘。")

    logger.info("任务完成。")


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="gather_local.py - 生成 data/logs/metadata.csv（扫描 local_dataset）")
    ap.add_argument("--root", type=str, default=str(RAW_LOCAL), help="local_dataset 根目录（默认 data/raw/local_dataset）")
    ap.add_argument("--phash-threshold", type=int, default=5, help="phash 汉明距离阈值判断近似重复（当安装 imagehash 时有效）")
    ap.add_argument("--dedup-action", choices=["ignore", "move"], default="ignore",
                    help="检测到重复后的动作：ignore(仅记录) 或 move(移动到 data/logs/duplicates/)")
    ap.add_argument("--move-duplicates", action="store_true", help="等同于 --dedup-action move（向后兼容）")
    ap.add_argument("--dry-run", action="store_true", help="只打印结果，不写 CSV、不移动文件")
    ap.add_argument("--verbose", action="store_true", help="打印更详细日志")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    # override ROOT if provided
    RAW = Path(args.root)
    # replace global RAW_LOCAL with passed root for one-run behavior
    RAW_LOCAL = RAW  # type: ignore
    main(args)
