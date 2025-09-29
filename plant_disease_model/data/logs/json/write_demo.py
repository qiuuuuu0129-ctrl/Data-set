# data/logs/json/write_demo.py
import json
from pathlib import Path

log_dir = Path(__file__).parent
log_file = log_dir / "example_log.json"

data = {"epoch": 1, "accuracy": 0.95, "loss": 0.123}

with open(log_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"日志已写入 {log_file}")
# 运行后会生成 data/logs/json/example_log.json