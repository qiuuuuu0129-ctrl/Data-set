# checkpoints/save_load_demo.py
import torch
import torch.nn as nn

# 简单示例模型
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = DummyNet()

# 保存模型
torch.save(model.state_dict(), "checkpoints/pytorch/dummy_model.pt")
print("模型已保存到 checkpoints/pytorch/dummy_model.pt")

# 加载模型
model2 = DummyNet()
model2.load_state_dict(torch.load("checkpoints/pytorch/dummy_model.pt"))
print("模型已加载成功")
