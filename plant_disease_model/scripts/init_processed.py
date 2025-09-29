from pathlib import Path

# 修改为顶层 processed 目录
root = Path("processed")
subdirs = ["train", "val", "test"]

# 创建主目录
root.mkdir(parents=True, exist_ok=True)

# 创建 README.md
readme_path = root / "README.md"
readme_content = """# Processed Dataset

此目录用于存放经过 dataset_builder.py 处理后的训练/验证/测试集数据。
"""
readme_path.write_text(readme_content, encoding="utf-8")

# 创建子目录及 __init__.py
for sd in subdirs:
    path = root / sd
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# 空文件，仅用于保持 Python package 结构\n", encoding="utf-8")

print("✅ processed 目录初始化完成")
