#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

class NetBlock(nn.Module):
    """
    Implementing the MetaNet approach
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self,in_channels, middle_channels, out_channels,V,U):
        super(NetBlock, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, feat_maps, metadata, V, U):
        metadata = torch.unsqueeze(metadata, -1)
        metadata = torch.unsqueeze(metadata, -1)
        x = self.metanet(metadata)
        x = x * feat_maps
        x = torch.sum(x,dim=1)
        # x = x.repeat(1, 8, 4)   #resnet網絡
        # x= x.repeat(1,98,4)     #vgg網絡
        x = x.repeat(1, 4, 4)   #densenet網絡
        t1 = self.fb(U)
        t2 = self.gb(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        x = x + V
        return x


