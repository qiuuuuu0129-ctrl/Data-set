"""
PyTorch 推理
"""
import torch
from torchvision import models
from src.datasets.transforms import build_transforms
from PIL import Image
import yaml

def infer_single_image(cfg_path: str, ckpt_path: str, img_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    transform = build_transforms(cfg["preprocess"], phase="test")
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # batch dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    # num_classes 可以从 metadata.csv 统计
    import pandas as pd
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    classes = sorted(df["plant"].unique())
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    print(f"Predicted: {classes[pred.item()]}")
