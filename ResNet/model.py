import torch.nn as nn
import torch


# This is for 18/34 layer
class BasicBlock(nn.Module):
    expansion = 1

    # downsaple表示捷径
    def __init__(self, in_channel, out_channnel, stride=1, dowmsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel=in_channel, out_channels=out_channnel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channnel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channnel, out_channels=out_channnel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channnel)

        self.downsample = dowmsample

    def forword(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# This is for 50/101/152 layer
class Bottleneck(nn.Module):
    expansion = 4  # 卷积核数量倍数

    # downsaple表示捷径
    def __init__(self, in_channel, out_channnel, stride=1, dowmsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel=in_channel, out_channels=out_channnel, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channnel)

        self.conv2 = nn.Conv2d(in_channels=out_channnel, out_channels=out_channnel, kernel_size=3, stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channnel)
        self.conv3 = nn.Conv2d(in_channels=out_channnel, out_channels=out_channnel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channnel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = dowmsample

    def forword(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # block:BasicBlock or Bottleneck
    # block_num:[3,4,6,3]
    # num_classes:训练集分类个数
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top  # 复杂增加，本次未使用
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = 1x1
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        if self.include_top:
            output = self.avgpool(output)
            output = nn.flatten(output, 1)
            output = self.fc(output)

        return output


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classer=num_classes, include_top=include_top)
