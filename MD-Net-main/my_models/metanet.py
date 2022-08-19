#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch

class MetaNet(nn.Module):
    """
    Implementing the MetaNet approach
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        metadata = torch.unsqueeze(metadata, -1)
        metadata = torch.unsqueeze(metadata, -1)
        x = self.metanet(metadata)
        # print('tttttttt',x.shape)
        # print('88888',feat_maps.shape)
        # feat_maps = torch.sum(feat_maps, dim=1)
        # feat_maps = feat_maps.unsqueeze(1)
        # feat_maps = feat_maps.repeat(1,32,1,1)
        # print(feat_maps.shape)
        x = x * feat_maps
        # x = torch.bmm(x,feat_maps)
        # print(x.shape)
        return x
