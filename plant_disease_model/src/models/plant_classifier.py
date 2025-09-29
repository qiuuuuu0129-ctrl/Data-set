# models/plant_classifier.py
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=3*64*64, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
# 这是一个非常简单的线性分类器示例，实际项目中会用更复杂的模型（如 ResNet）
# 你可以在 train.py / evaluate.py / infer.py 中导入并使用这个模型
# from src.models.plant_classifier import SimpleClassifier
# 例如：
# model = SimpleClassifier(input_dim=3*224*224, num_classes=train_dataset.num_classes())
# model.to(device)
# 当然，实际项目中更推荐使用 torchvision.models 里的预训练模型，如 res
# net18、resnet50 等，并根据你的分类任务调整最后的全连接层。
# 例如：
# from torchvision import models
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, train_dataset.num_classes())
# model.to(device)
# 这样可以利用预训练模型在大规模数据集上学到的特征
# 你可以根据需要扩展这个文件，添加更多模型定义
# 例如添加 CNN、Transformer 等模型
# 记得在 train.py / evaluate.py / infer.py 中相应地导入和使用新模型
# 另外，确保在保存和加载模型时使用 model.state_dict() 和 load_state_dict()
#
