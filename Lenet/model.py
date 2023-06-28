import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # 网络层结构
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # （输入深度，输出深度，卷积核大小）
        self.poo1 = nn.MaxPool2d(2, 2)  # 池化核大小和步距
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.poo2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # (全连接层输入，输出）
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 正向传播过程，x是输入数据[b,c,h,w]
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.poo1(x)
        x = F.relu(self.conv2(x))
        x = self.poo2
        # view将其展平为一维向量，再送入全连接层，-1表示自动推理输入维度
        x = F.view(-1.32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3
        return x
