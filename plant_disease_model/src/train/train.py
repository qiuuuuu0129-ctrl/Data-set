"""
自动训练主程序
- 支持 classification (plant / plant+disease)
- 支持选择 backbone, lr, batch, epochs
- 输出 checkpoint 到 checkpoints/pytorch/
"""
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import models
from tqdm import tqdm

from src.datasets.dataset import PlantDiseaseDataset
from src.datasets.transforms import build_transforms
import yaml

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(cfg_path: str):
    cfg = load_cfg(cfg_path)
    train_trans = build_transforms(cfg["preprocess"], phase="train", image_size=224)
    val_trans = build_transforms(cfg["preprocess"], phase="val", image_size=224)

    train_dataset = PlantDiseaseDataset(metadata_csv=cfg["data"]["metadata_csv"], split="train", transform=train_trans)
    val_dataset = PlantDiseaseDataset(metadata_csv=cfg["data"]["metadata_csv"], split="val", transform=val_trans)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(train_dataset.num_classes()).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    epochs = cfg["train"]["epochs"]
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Val Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            ckpt_dir = Path("checkpoints/pytorch")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print("Saved best model checkpoint.")
