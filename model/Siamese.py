# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

from model.getFeature import LeNet5


class BiSiamese(nn.Module):
    def __init__(self, num_class=2):
        super(BiSiamese, self).__init__()
        self.get_feature = LeNet5(False)
        self.fc1 = None
        self.bn1 = nn.BatchNorm1d(84)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_class)
        self.bn2 = nn.BatchNorm1d(num_class)

        self.act2 = nn.Sigmoid()

    def forward(self, images):
        x1, x2 = images[0], images[1]
        out1 = self.get_feature.forward(x1)
        out2 = self.get_feature.forward(x2)
        out = torch.cat((out1, out2), dim=1)
        in_feature = out.shape[-1]
        if self.fc1 is None:
            device = next(self.get_feature.parameters()).device
            self.fc1 = nn.Linear(in_feature, 84).to(device)
        out = self.act1(self.bn1(self.fc1(out)))
        out = self.act2(self.bn2(self.fc2(out)))
        out = self.act2(out)
        return out


class TriSiamese(nn.Module):
    def __init__(self):
        super(TriSiamese, self).__init__()
        self.get_feature = LeNet5(False)
        self.get_vector = None

    def forward(self, images):  # positive,anchor,negative
        xa = images[0]
        if len(images) == 3:
            xn = images[2]
            xp = images[1]
            outp = self.get_feature.forward(xp)
            outa = self.get_feature.forward(xa)
            outn = self.get_feature.forward(xn)
            if self.get_vector is None:
                in_feature = outp.shape[-1]
                device = next(self.get_feature.parameters()).device
                self.get_vector = nn.Linear(in_feature, 84).to(device)
            outp = self.get_vector(outp)
            outa = self.get_vector(outa)
            outn = self.get_vector(outn)
            out = torch.stack((outa, outp, outn))
        elif len(images) == 2:
            xp = images[1]
            outp = self.get_feature.forward(xp)
            outa = self.get_feature.forward(xa)
            if self.get_vector is None:
                in_feature = outp.shape[-1]
                device = next(self.get_feature.parameters()).device
                self.get_vector = nn.Linear(in_feature, 84).to(device)
            outp = self.get_vector(outp)
            outa = self.get_vector(outa)
            out = torch.stack((outa, outp))
        else:
            xa = images
            outa = self.get_feature.forward(xa)
            if self.get_vector is None:
                in_feature = outa.shape[-1]
                device = next(self.get_feature.parameters()).device
                self.get_vector = nn.Linear(in_feature, 84).to(device)
            outa = self.get_vector(outa)
            out = torch.stack((outa,))
        return out
