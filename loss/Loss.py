# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# 后来发现pytorch实现了TripletLoss，所以这个Loss就没用了
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs):
        positive, anchor, negative = outputs[0], outputs[1], outputs[2]
        if len(anchor.shape) == 1:  # 如果输入是一维向量（无batch操作）
            distance_positive = (anchor - positive).pow(2).sum()  # 不指定维度进行求和
            distance_negative = (anchor - negative).pow(2).sum()  # 不指定维度进行求和
        else:  # 如果输入是二维向量（有batch操作）
            distance_positive = (anchor - positive).pow(2).sum(1)  # 在每个样本内进行求和
            distance_negative = (anchor - negative).pow(2).sum(1)  # 在每个样本内进行求和
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
