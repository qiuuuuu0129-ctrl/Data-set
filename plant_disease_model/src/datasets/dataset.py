"""
src/datasets/dataset.py

PlantDiseaseDataset - 将 data/logs/metadata.csv 与 transforms 结合成 PyTorch Dataset。

功能：
- 从 metadata.csv 读取样本信息（filepath, plant, disease, optional: split）
- 支持由 "plant" 字段自动构造类别映射（可选择按 "plant+disease" 组合）
- 支持直接传入 csv_file 列表（如果你用 dataset_builder 只生成 csv 列表）
- 支持可选 transform（来自 src.datasets.transforms）
- 支持返回原始相对路径、PIL 图像、tensor、label id、label name、metadata 字典
- 简单的 class_balance 统计辅助函数

依赖：torch, torchvision, pandas, pillow
"""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset

# default project root detection (可按需覆盖)
PROJECT_ROOT = Path.cwd()
DEFAULT_METADATA = PROJECT_ROOT / "data" / "logs" / "metadata.csv"


class PlantDiseaseDataset(Dataset):
    def __init__(self,
                 metadata_csv: Optional[str] = None,
                 root_dir: Optional[str] = None,
                 split: Optional[str] = None,
                 label_by: str = "plant",  # or "plant+disease"
                 transform = None,
                 loader = None,
                 allowed_exts: Optional[List[str]] = None):
        """
        Args:
            metadata_csv: path to metadata.csv. If None uses default data/logs/metadata.csv
            root_dir: project root for resolving relative filepaths in metadata (defaults to cwd)
            split: if metadata has a 'split' column (train/val/test), pass "train"/"val"/"test" to filter
            label_by: "plant" (default) or "plant+disease" to create label categories
            transform: torchvision transforms to apply
            loader: function(path) -> PIL.Image (if you want custom loader)
            allowed_exts: optional list of allowed extensions to filter
        """
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT
        self.metadata_csv = Path(metadata_csv) if metadata_csv else DEFAULT_METADATA
        self.transform = transform
        self.loader = loader if loader is not None else self.default_loader
        self.label_by = label_by
        self.allowed_exts = [e.lower() for e in allowed_exts] if allowed_exts else None

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"metadata csv not found: {self.metadata_csv}")

        self.df = pd.read_csv(self.metadata_csv)

        # optional split filter
        if split and "split" in self.df.columns:
            self.df = self.df[self.df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)

        # Filter files missing or with disallowed ext
        self.df["filepath_abs"] = self.df["filepath"].apply(lambda p: (self.root_dir / p) if not Path(p).is_absolute() else Path(p))
        exists_mask = self.df["filepath_abs"].apply(lambda p: p.exists())
        if not exists_mask.all():
            missing = self.df.loc[~exists_mask, "filepath"].tolist()
            # warn (but keep only existing)
            print(f"[warn] {len(missing)} files listed in metadata not found on disk. They will be ignored.")
            self.df = self.df.loc[exists_mask].reset_index(drop=True)

        if self.allowed_exts:
            self.df = self.df[self.df["filepath"].str.lower().apply(lambda s: any(s.endswith(ext) for ext in self.allowed_exts))]

        # build label mapping
        if self.label_by == "plant":
            label_col = "plant"
        elif self.label_by in ("plant+disease", "plant_disease", "plant+disease"):
            # ensure 'disease' column exists (if not, fallback to plant)
            if "disease" not in self.df.columns:
                print("[warn] 'disease' column missing in metadata; falling back to 'plant'.")
                label_col = "plant"
            else:
                # create new temporary combined label column
                self.df["plant_disease"] = self.df["plant"].astype(str) + "__" + self.df["disease"].astype(str)
                label_col = "plant_disease"
        else:
            raise ValueError("label_by must be 'plant' or 'plant+disease'")

        self.label_col = label_col
        label_names = sorted(self.df[self.label_col].unique().tolist())
        self.label2id = {name: idx for idx, name in enumerate(label_names)}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # prepare samples list
        self.samples = []
        for _, row in self.df.iterrows():
            fp = Path(row["filepath_abs"])
            label_name = row[self.label_col]
            label_id = self.label2id[label_name]
            meta = row.to_dict()
            self.samples.append({
                "filepath": fp,
                "label": label_id,
                "label_name": label_name,
                "meta": meta
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        path: Path = s["filepath"]
        try:
            img = self.loader(path)
        except UnidentifiedImageError:
            # return a dummy (black) image to avoid crash; still return metadata to inspect
            print(f"[warn] cannot open image {path}, returning black image.")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = s["label"]
        label_name = s["label_name"]
        meta = s["meta"]

        return img, label, label_name, meta

    @staticmethod
    def default_loader(path: Path):
        return Image.open(path).convert("RGB")

    def get_class_balance(self) -> Dict[str, int]:
        counts = {}
        for s in self.samples:
            counts[s["label_name"]] = counts.get(s["label_name"], 0) + 1
        return counts

    def num_classes(self) -> int:
        return len(self.label2id)

    def make_subset_csv(self, out_csv: str):
        """Export the subset of metadata used by this dataset to csv (useful for reproducibility)."""
        df = pd.DataFrame([s["meta"] for s in self.samples])
        df.to_csv(out_csv, index=False)
        print(f"Subset metadata saved to {out_csv}")

# optional collate_fn (if needed)
def default_collate_batch(batch):
    """
    batch: list of tuples (img_tensor or PIL, label, label_name, meta)
    Convert images to tensors if PIL and stack. Assumes images are torch tensors already after transform.
    """
    imgs = []
    labels = []
    metas = []
    for item in batch:
        imgs.append(item[0])
        labels.append(item[1])
        metas.append(item[3])
    # if PIL images slipped through convert them
    if isinstance(imgs[0], Image.Image):
        import torchvision.transforms as T
        imgs = [T.ToTensor()(im) for im in imgs]
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels, metas
