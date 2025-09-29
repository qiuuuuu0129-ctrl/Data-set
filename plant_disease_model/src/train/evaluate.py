"""
模型评估脚本
- 可计算 accuracy / F1 / per-class stats
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.datasets.dataset import PlantDiseaseDataset
from src.datasets.transforms import build_transforms
from torchvision import models
import yaml
from pathlib import Path

def evaluate(cfg_path: str, ckpt_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    val_trans = build_transforms(cfg["preprocess"], phase="val", image_size=224)
    val_dataset = PlantDiseaseDataset(metadata_csv=cfg["data"]["metadata_csv"], split="val", transform=val_trans)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, val_dataset.num_classes())
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Val Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[val_dataset.id2label[i] for i in range(val_dataset.num_classes())]))
