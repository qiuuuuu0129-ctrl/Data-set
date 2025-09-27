#!/usr/bin/env python3
"""
capture_dataset_full.py
专业版采集脚本（读取 config.yaml）：
- 图像质量检测（模糊、亮度）
- 人脸检测跳过
- 会话内 phash 去重（避免重复保存）
- 保存 metadata CSV + JSON 日志
- 可选后台上传（requests）
- 交互式输入标签或批量 labels（config 可扩展）

使用： python capture_dataset_full.py
"""

import cv2
import os
import time
import yaml
import json
import hashlib
import requests
import threading
import numpy as np
from PIL import Image
import imagehash
from datetime import datetime
from tqdm import tqdm
from collections import deque

# 这里是一些工具函数
# ------------------- utils -------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def compute_sha256_from_bytes(b: bytes):
    import hashlib
    return hashlib.sha256(b).hexdigest()

def pil_image_from_bgr(bgr_img):
    # BGR (cv2) -> RGB PIL
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def compute_phash(pil_img):
    return str(imagehash.phash(pil_img))

def save_image_jpg(path, bgr_img, quality=95):
    # 保存并返回写入文件的 bytes
    # 使用 cv2.imencode 获得 bytes，同时写文件
    ok, buf = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image")
    b = buf.tobytes()
    with open(path, 'wb') as f:
        f.write(b)
    return b

# ------------------- detectors -------------------
def load_face_cascade():
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return cascade
    except Exception:
        return None

def detect_face(face_cascade, gray_img, min_size):
    if face_cascade is None:
        return False
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=tuple(min_size))
    return len(faces) > 0

# ------------------- uploader -------------------
class BackgroundUploader:
    def __init__(self, url, timeout, retries, max_threads=2):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pool = deque()
        self.lock = threading.Lock()
        self.max_threads = max_threads

    def submit(self, filepath):
        # spawn thread if pool < max_threads
        t = threading.Thread(target=self._upload_with_retry, args=(filepath,))
        t.daemon = True
        t.start()
        with self.lock:
            self.pool.append(t)

    def _upload_with_retry(self, filepath):
        for attempt in range(1, self.retries + 2):
            try:
                files = {'file': open(filepath, 'rb')}
                r = requests.post(self.url, files=files, timeout=self.timeout)
                if r.status_code == 200:
                    print(f"🔼 上传成功: {os.path.basename(filepath)}")
                    return True
                else:
                    print(f"⚠️ 上传失败 HTTP {r.status_code} (尝试 {attempt})")
            except Exception as e:
                print(f"⚠️ 上传异常: {e} (尝试 {attempt})")
            time.sleep(1)
        print(f"❌ 上传重试失败: {filepath}")
        return False

# ------------------- main capture logic -------------------
def capture_label_session(cfg, label, uploader=None):
    camera_index = cfg.get("camera_index", 0)
    num_images = int(cfg.get("num_images", 50))
    interval_sec = float(cfg.get("interval_sec", 0.5))
    max_frames = int(cfg.get("max_frames", 1000))
    blur_threshold = float(cfg.get("blur_threshold", 100.0))
    brightness_low = float(cfg.get("brightness_low", 40))
    brightness_high = float(cfg.get("brightness_high", 220))
    enable_phash = bool(cfg.get("enable_phash_check", True))
    phash_thresh = int(cfg.get("phash_hamming_threshold", 6))
    detect_face_flag = bool(cfg.get("detect_face", True))
    face_min_size = cfg.get("face_min_size", [50,50])
    show_preview = bool(cfg.get("show_preview_window", True))

    # prepare dirs
    base_dir = cfg.get("data_dir", "data")
    logs_dir = cfg.get("logs_dir", "logs")
    create_dir(base_dir); create_dir(logs_dir); create_dir(cfg.get("log_json_dir","logs/json"))

    date_subdir = datetime.now().strftime("%Y%m%d_%H%M%S") if cfg.get("per_label_dir_date", True) else ""
    save_dir = os.path.join(base_dir, label, date_subdir) if date_subdir else os.path.join(base_dir, label)
    create_dir(save_dir)

    metadata_csv = cfg.get("metadata_csv", os.path.join(logs_dir, "metadata.csv"))
    create_dir(os.path.dirname(metadata_csv))

    face_cascade = load_face_cascade() if detect_face_flag else None

    # in-session phash set (to avoid saving near-duplicated frames)
    phash_set = []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 index=", camera_index)
        return

    print(f"\n🌱 开始采集类别 '{label}' 到 {save_dir}，目标有效图片 {num_images} 张")
    saved = 0
    frames = 0
    session_log = {"label": label, "start": datetime.now().isoformat(), "entries": []}
    pbar = tqdm(total=num_images, desc=f"采集 {label}", unit="img")

    while saved < num_images and frames < max_frames:
        ret, frame = cap.read()
        frames += 1
        if not ret:
            time.sleep(0.1)
            continue

        # preview
        if show_preview:
            cv2.imshow("Preview", frame)
            # allow keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord(cfg.get("keyboard_quit_key","q")):
                print("⏹ 用户提前退出当前类别采集")
                break
            if key == ord(cfg.get("keyboard_skip_key","n")):
                print("➡️ 用户跳过当前类别")
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # checks
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < blur_threshold:
            session_log["entries"].append({"frame": frames, "action":"skip","reason":"blurry","laplacian":laplacian_var})
            # skip frame (do not count toward num_images)
            continue

        mean_brightness = float(np.mean(gray))
        if mean_brightness < brightness_low or mean_brightness > brightness_high:
            session_log["entries"].append({"frame": frames, "action":"skip","reason":"lighting","brightness":mean_brightness})
            continue

        if detect_face_flag and detect_face(face_cascade, gray, face_min_size):
            session_log["entries"].append({"frame": frames, "action":"skip","reason":"face_detected"})
            continue

        # compute phash and duplication check (convert BGR->PIL)
        pil_img = pil_image_from_bgr(frame)
        phash = compute_phash(pil_img) if enable_phash else None
        if enable_phash:
            is_dup = False
            for existing in phash_set:
                # compute Hamming distance between hex strings
                # imagehash hex string -> imagehash.ImageHash: using imagehash.hex_to_hash
                d = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing)
                if d <= phash_thresh:
                    is_dup = True
                    break
            if is_dup:
                session_log["entries"].append({"frame": frames, "action":"skip","reason":"duplicate","phash":phash})
                continue

        # passed all checks, save image
        fname = f"{label}_{saved+1:04d}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        fpath = os.path.join(save_dir, fname)
        try:
            # save as jpg and compute sha256
            jpg_bytes = save_image_jpg(fpath, frame, quality=95)
            sha256 = compute_sha256_from_bytes(jpg_bytes)
            width = int(frame.shape[1]); height = int(frame.shape[0])
            # record metadata
            row = {
                "filepath": fpath,
                "label": label,
                "timestamp": datetime.now().isoformat(),
                "sha256": sha256,
                "phash": phash,
                "width": width,
                "height": height,
                "laplacian": laplacian_var,
                "brightness": mean_brightness
            }
            # append to csv (create if not exists)
            write_metadata_row(metadata_csv, row)
            session_log["entries"].append({"frame": frames, "action":"save","meta":row})
            saved += 1
            if enable_phash and phash is not None:
                phash_set.append(phash)
            pbar.update(1)
            print(f"✅ 保存 {saved}/{num_images}: {fpath}")
        except Exception as e:
            print("❌ 保存失败：", e)
            session_log["entries"].append({"frame": frames, "action":"error","error":str(e)})
        # optional upload in background
        if uploader:
            uploader.submit(fpath)

        # respect interval
        if interval_sec > 0:
            time.sleep(interval_sec)

    pbar.close()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    session_log["end"] = datetime.now().isoformat()
    session_log["saved"] = saved
    session_log["frames"] = frames
    # save session log json
    json_path = os.path.join(cfg.get("log_json_dir","logs/json"), f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    create_dir(os.path.dirname(json_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_log, f, ensure_ascii=False, indent=2)

    print(f"📸 会话结束：保存 {saved} 张图片，日志保存在 {json_path}")
    return session_log

# helper to append metadata CSV
def write_metadata_row(csv_path, row: dict):
    header = ["filepath","label","timestamp","sha256","phash","width","height","laplacian","brightness"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        # escape commas in filepath? we'll assume none
        vals = [str(row.get(h,"")) for h in header]
        f.write(",".join(vals) + "\n")

# ------------------- main -------------------
def main():
    cfg = load_config("config.yaml")

    uploader = None
    if cfg.get("upload",{}).get("enable", False):
        up_cfg = cfg["upload"]
        uploader = BackgroundUploader(up_cfg["url"], up_cfg.get("timeout_sec",8), up_cfg.get("retries",2), up_cfg.get("background_threads",2))

    print("🌿 PlantAI 专业采集工具 (capture_dataset_full)")
    print("配置文件加载自 config.yaml")
    # optional: accept comma-separated labels from user or batch mode
    batch = input("是否输入多个类别一次性采集？(y/n, 默认 n): ").strip().lower()
    if batch == "y":
        labels_line = input("请输入类别列表，英文逗号分隔 (例如 rose,sunflower,tulip): ").strip()
        labels = [l.strip() for l in labels_line.split(",") if l.strip()]
    else:
        labels = []
        while True:
            lab = input("输入类别名称 (输入 exit 退出)：").strip()
            if lab.lower() == "exit":
                break
            if lab == "":
                continue
            labels.append(lab)
            nxt = input("继续添加？(y/n, 默认 n): ").strip().lower()
            if nxt != "y":
                break

    if not labels:
        print("未指明任何类别，退出。")
        return

    for label in labels:
        print(f"\n>>> 开始类别: {label}")
        capture_label_session(cfg, label, uploader=uploader)

    print("全部采集任务完成。")  #test工具函数

if __name__ == "__main__":
    main()