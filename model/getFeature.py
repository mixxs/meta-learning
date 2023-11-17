# -*- coding: utf-8 -*-
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, isTrain):
        super(LeNet5, self).__init__()
        self.isTrain = isTrain
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.bn5 = nn.BatchNorm1d(10)

    def forward(self, x):  # 输入形状 : N，1，28，28
        x = self.act1(self.bn1(self.conv1(x)))  # 输出：N，6，28，28
        x = self.pool1(x)  # 输出N，6，14，14
        x = self.act1(self.bn2(self.conv2(x)))  # 输出 N，16，10,10
        x = self.pool2(x)  # 输出 N，16，5，5
        x = x.view(-1, 16 * 5 * 5)  # 输出 N，400
        x = self.bn3(self.act1(self.fc1(x)))  # 输出N，120
        if self.isTrain:
            x = self.bn4(self.act1(self.fc2(x)))  # 输出N，84
            x = self.bn5(self.fc3(x))  # 输出N，10。使用CE loss会进行softmax，所以输出层就不加激活了
        return x
