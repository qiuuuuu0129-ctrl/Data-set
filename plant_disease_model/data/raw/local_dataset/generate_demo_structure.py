# data/raw/local_dataset/generate_demo_structure.py
from pathlib import Path

plants = ["tomato", "potato"]
base_dir = Path(__file__).parent

for plant in plants:
    p_dir = base_dir / plant
    p_dir.mkdir(exist_ok=True, parents=True)
    # 生成占位文件
    (p_dir / ".gitkeep").touch()

print(f"生成了示例数据集结构: {list(base_dir.iterdir())}")
