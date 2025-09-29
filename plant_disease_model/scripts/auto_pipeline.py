"""
一键运行完整流水线：
gather_local -> download_external -> preprocess -> build dataset -> train -> eval
"""
import subprocess
import sys
import yaml
from pathlib import Path

def run_pipeline(cfg_path="configs/dataset_config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # step1: 本地采集数据或已有数据
    subprocess.run([sys.executable, "src/data_collection/gather_local.py"], check=True)

    # step2: 下载外部数据
    subprocess.run([sys.executable, "src/data_collection/download_external.py"], check=True)

    # step3: preprocess
    from src.data_collection.preprocess import preprocess_dataset
    preprocess_dataset(cfg["data"]["raw_local_path"], cfg["data"]["processed_path"])
    preprocess_dataset(cfg["data"]["raw_external_path"], cfg["data"]["processed_path"])

    # step4: dataset_builder
    subprocess.run([sys.executable, "src/data_collection/dataset_builder.py", "--config", cfg_path], check=True)

    # step5: train
    subprocess.run([sys.executable, "src/train/train.py", "--config", cfg_path], check=True)

    # step6: evaluate
    ckpt = Path("checkpoints/pytorch/best_model.pth")
    if ckpt.exists():
        subprocess.run([sys.executable, "src/train/evaluate.py", "--config", cfg_path, "--ckpt", str(ckpt)], check=True)

if __name__ == "__main__":
    run_pipeline()
