import torch.utils.data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


def main():
    # 数据转换，其中有转换到张量，标准化处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train为True会下载训练集
    # 5w训练图像，10类
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 加载一批数据，shuffle是打乱数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
    # 1w张测试图像
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # 测试集直接用1w图像,而且不需要打乱数据
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    test_data_iter = iter(testloader)
    test_image, test_label = test_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    # 交叉熵损失函数，用于分类任务，包含了softmax
    loss_function = nn.CrossEntropyLoss()
    # 优化器，LeNet所有参数，学习率
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(5):
        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
            # data is a list of [inputs,labels]
            inputs, labels = data

            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            # 参数更新
            optimizer.step()
            # print
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)  # [batch,10]
                    # dim=1是因为0代表的是batch，[1]是只知道索引位置
                    predict_y = torch.max(outputs, dim=1)[1]
                    # .sum()后是Tensor格式求和，item()将结构转换为标量
                    accuracy = (predict_y == test_label).sum().item() / test_label.size()
                    print('[%d, %5d] train_loss:%.3f test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training !')
    save_path = './LeNet.pth'
    torch.save(net.state_dict(),save_path)