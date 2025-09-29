"""
下载和整理外部开源数据集
- 可配置 gdown / kaggle API / URL 列表
- 自动放到 data/raw/external_dataset/<label>/
"""
import os
from pathlib import Path
import gdown
import zipfile
import tarfile
import shutil

OUTPUT_DIR = Path("data/raw/external_dataset")

def download_from_gdrive(url: str, out_dir: Path = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    gdown.download(url, str(out_dir / "dataset.zip"), quiet=False)
    # 解压
    with zipfile.ZipFile(out_dir / "dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    os.remove(out_dir / "dataset.zip")
    print("Download and extraction done.")

def download_from_tar(url: str, out_dir: Path = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    import requests
    local_path = out_dir / "dataset.tar.gz"
    r = requests.get(url, stream=True)
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # 解压
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    os.remove(local_path)
    print("Download and extraction done.")

if __name__ == "__main__":
    # 示例调用
    gdrive_url = "YOUR_PUBLIC_GDRIVE_LINK"
    download_from_gdrive(gdrive_url)

