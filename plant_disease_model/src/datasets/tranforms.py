#它是一个可配置、可复用、对 PyTorch-friendly 的 transforms 工厂，可以根据你在 
#configs/dataset_config.yaml 里写的 preprocess / augmentation 配置动态构建 train / val / test 
#的 transform pipelines。代码包含常用变换（resize/short-side 保持长宽比、center/crop、随机翻转、
#随机旋转、color jitter、随机擦除/ Cutout、自定义 ToTensor & Normalize）并且易于扩展

"""
src/datasets/transforms.py

可配置的 transforms 工厂 (PyTorch / torchvision 风格)。

主要接口：
    build_transforms(cfg: dict, phase: str) -> torchvision.transforms.Compose

cfg 期望结构（可参考 configs/dataset_config.yaml）:
cfg = {
    "preprocess": {
        "resize": {"enabled": True, "short_side": 256, "maintain_aspect": True},
        "normalize": {"enabled": True, "mean": [...], "std": [...]},
        "augmentation": {
            "train": [
                {"name": "random_flip", "prob": 0.5},
                {"name": "random_rotation", "max_deg": 15, "prob": 0.3},
                {"name": "color_jitter", "brightness": 0.2, ... , "prob": 0.3},
                {"name": "random_erasing", "prob": 0.2, "scale": [0.02, 0.3]}
            ]
        }
    }
}

示例用法：
    from src.datasets.transforms import build_transforms
    train_t = build_transforms(cfg, "train")
    val_t = build_transforms(cfg, "val")
"""
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import math

try:
    import torch
    from torchvision import transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    raise RuntimeError("torch and torchvision required for this transforms module")

# ---------------------------
# Small custom transforms
# ---------------------------
class ShortSideResize:
    """
    Resize so that the short side == short_side, keep aspect ratio.
    If short_side is None or <=0, skip.
    """
    def __init__(self, short_side: int):
        assert isinstance(short_side, int) and short_side > 0
        self.short_side = short_side

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size  # PIL: (width, height)
        if w <= 0 or h <= 0:
            return img
        if min(w, h) == self.short_side:
            return img
        if w < h:
            new_w = self.short_side
            new_h = int(math.ceil(h * (new_w / w)))
        else:
            new_h = self.short_side
            new_w = int(math.ceil(w * (new_h / h)))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)

    def __repr__(self):
        return f"ShortSideResize(short_side={self.short_side})"


class CenterCropIfNeeded:
    """
    Center crop to (size, size) if both dims >= size.
    If maintain_aspect resize has been used, this is a typical choice for val/test.
    """
    def __init__(self, size: int):
        assert isinstance(size, int) and size > 0
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return T.functional.center_crop(img, self.size)

    def __repr__(self):
        return f"CenterCropIfNeeded(size={self.size})"


class RandomCutout:
    """
    Simple Cutout implementation: random square holes.
    args:
        num_holes: number of holes
        max_h_size, max_w_size: maximum hole size in pixels (if None uses size*0.5)
        prob: probability to apply
    """
    def __init__(self, num_holes: int = 1, max_h_size: Optional[int] = None, max_w_size: Optional[int] = None, prob: float = 0.5):
        self.num_holes = num_holes
        self.max_h = max_h_size
        self.max_w = max_w_size
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() > self.prob:
            return img
        img_tensor = T.functional.to_tensor(img)  # C,H,W
        c, h, w = img_tensor.shape
        max_h = self.max_h or int(h * 0.25)
        max_w = self.max_w or int(w * 0.25)
        for _ in range(self.num_holes):
            hh = random.randint(1, max(1, max_h))
            ww = random.randint(1, max(1, max_w))
            top = random.randint(0, max(0, h - hh))
            left = random.randint(0, max(0, w - ww))
            img_tensor[:, top:top+hh, left:left+ww] = 0.0
        img = T.functional.to_pil_image(img_tensor)
        return img

    def __repr__(self):
        return f"RandomCutout(num_holes={self.num_holes}, max_h={self.max_h}, max_w={self.max_w}, prob={self.prob})"


# ---------------------------
# Transform builder helpers
# ---------------------------
def _build_resize(cfg: Dict) -> Optional[Any]:
    r_cfg = cfg.get("resize", {}) if cfg else {}
    if not r_cfg.get("enabled", False):
        return None
    short_side = r_cfg.get("short_side", None)
    maintain = r_cfg.get("maintain_aspect", True)
    if maintain and short_side:
        return ShortSideResize(int(short_side))
    else:
        # fallback: simple resize to square short_side x short_side if provided
        if short_side:
            return T.Resize((short_side, short_side), interpolation=Image.BILINEAR)
        return None


def _build_normalize(cfg: Dict) -> Optional[T.Normalize]:
    n_cfg = cfg.get("normalize", {}) if cfg else {}
    if not n_cfg.get("enabled", False):
        return None
    mean = n_cfg.get("mean", [0.485, 0.456, 0.406])
    std = n_cfg.get("std",  [0.229, 0.224, 0.225])
    return T.Normalize(mean=mean, std=std)


def _parse_aug_spec(spec: Dict) -> Optional[Any]:
    """
    spec: {"name": "random_flip", "prob": 0.5}
    returns a torchvision/PIL transform or None
    """
    name = spec.get("name", "").lower()
    prob = float(spec.get("prob", 1.0))

    if name in ("random_flip", "random_horizontal_flip"):
        p = spec.get("prob", 0.5)
        return T.RandomHorizontalFlip(p=p)
    if name in ("random_vertical_flip",):
        p = spec.get("prob", 0.5)
        return T.RandomVerticalFlip(p=p)
    if name in ("random_rotation", "rotation"):
        max_deg = float(spec.get("max_deg", 15.0))
        p = spec.get("prob", 1.0)
        # wrap in RandomApply if p < 1
        rot = T.RandomRotation(degrees=max_deg)
        return T.RandomApply([rot], p=p) if p < 1.0 else rot
    if name in ("color_jitter",):
        cj_args = {}
        for k in ("brightness", "contrast", "saturation", "hue"):
            if k in spec:
                cj_args[k] = float(spec[k])
        cj = T.ColorJitter(**cj_args) if cj_args else None
        p = spec.get("prob", 1.0)
        if cj is None:
            return None
        return T.RandomApply([cj], p=p) if p < 1.0 else cj
    if name in ("random_resized_crop",):
        size = int(spec.get("size", 224))
        scale = tuple(spec.get("scale", (0.08, 1.0)))
        ratio = tuple(spec.get("ratio", (3. / 4., 4. / 3.)))
        p = spec.get("prob", 1.0)
        rrc = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
        return T.RandomApply([rrc], p=p) if p < 1.0 else rrc
    if name in ("random_crop",):
        size = int(spec.get("size", 224))
        p = spec.get("prob", 1.0)
        rc = T.RandomCrop(size)
        return T.RandomApply([rc], p=p) if p < 1.0 else rc
    if name in ("random_erasing", "random_erase"):
        # torchvision RandomErasing works on tensors AFTER ToTensor()
        p = float(spec.get("prob", 0.25))
        scale = tuple(spec.get("scale", (0.02, 0.33)))
        ratio = tuple(spec.get("ratio", (0.3, 3.3)))
        value = spec.get("value", 0)  # 0 or mean
        return T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)
    if name in ("cutout", "random_cutout"):
        num_holes = int(spec.get("num_holes", 1))
        max_h = spec.get("max_h_size", None)
        max_w = spec.get("max_w_size", None)
        p = float(spec.get("prob", 0.5))
        return RandomCutout(num_holes=num_holes, max_h_size=max_h, max_w_size=max_w, prob=p)
    # unknown augmentation
    return None


def _build_augmentation_pipeline(aug_cfg_list: List[Dict]) -> List[Any]:
    """Given list of augmentation specs return list of transforms (PIL-based)"""
    out = []
    for spec in (aug_cfg_list or []):
        t = _parse_aug_spec(spec)
        if t is None:
            continue
        out.append(t)
    return out


# ---------------------------
# Public API
# ---------------------------
def build_transforms(cfg: Dict, phase: str = "train", image_size: int = 224) -> T.Compose:
    """
    Build torchvision transforms based on cfg and phase ('train'|'val'|'test').
    Returns torchvision.transforms.Compose ready to use in Dataset.
    """
    assert phase in ("train", "val", "test")
    preprocess_cfg = cfg.get("preprocess", {}) if cfg else {}

    resize_transform = _build_resize(preprocess_cfg)
    normalize_transform = _build_normalize(preprocess_cfg)

    transform_list: List[Any] = []

    # Pre-resize (short side)
    if resize_transform is not None:
        transform_list.append(resize_transform)

    # For train: parse augmentation list under preprocess.augmentation.train
    if phase == "train":
        aug_list = preprocess_cfg.get("augmentation", {}).get("train", [])
        aug_transforms = _build_augmentation_pipeline(aug_list)
        transform_list.extend(aug_transforms)

        # ensure final random crop to image_size for training (if not provided)
        has_crop = any(isinstance(t, (T.RandomResizedCrop, T.RandomCrop)) for t in transform_list)
        if not has_crop:
            transform_list.append(T.RandomResizedCrop(image_size))
    else:
        # val/test: center crop to image_size if short-side resize used,
        # else resize to (image_size, image_size)
        if resize_transform is not None:
            transform_list.append(CenterCropIfNeeded(image_size))
        else:
            transform_list.append(T.Resize((image_size, image_size)))

    # ToTensor
    transform_list.append(T.ToTensor())

    # Normalize (on tensors)
    if normalize_transform is not None:
        transform_list.append(normalize_transform)

    # For val/test optionally add RandomErasing? usually no.
    composed = T.Compose(transform_list)
    return composed


# ---------------------------
# Small helper to convert dataset_config style to builder call
# ---------------------------
def from_config(cfg: Dict, phase: str = "train", image_size: int = 224) -> T.Compose:
    """
    Convenience wrapper to accept either full config dict or the 'preprocess' dict.
    """
    if "preprocess" in cfg:
        return build_transforms(cfg, phase=phase, image_size=image_size)
    else:
        # assume cfg is already preprocess sub-dict
        return build_transforms({"preprocess": cfg}, phase=phase, image_size=image_size)


# ---------------------------
# Example quick test (not executed on import)
# ---------------------------
if __name__ == "__main__":
    # quick smoke example
    example_cfg = {
        "preprocess": {
            "resize": {"enabled": True, "short_side": 256, "maintain_aspect": True},
            "normalize": {"enabled": True, "mean": [0.485,0.456,0.406], "std":[0.229,0.224,0.225]},
            "augmentation": {
                "train": [
                    {"name": "random_flip", "prob": 0.5},
                    {"name": "color_jitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "prob": 0.3},
                    {"name": "random_rotation", "max_deg": 10, "prob": 0.2},
                    {"name": "cutout", "num_holes": 1, "max_h_size": 64, "max_w_size": 64, "prob": 0.3}
                ]
            }
        }
    }

    t = build_transforms(example_cfg, phase="train", image_size=224)
    print("Train pipeline:", t)
    t2 = build_transforms(example_cfg, phase="val", image_size=224)
    print("Val pipeline:", t2)
