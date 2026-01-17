""" Pytorch implementation of modified HRNet where the input data is not in the same scale as PROBA V data"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class PerChannelAffine(nn.Module):
    def __init__(self, channels: int, init_scale=1.0, init_bias=0.0, force_positive_scale=True):
        super().__init__()
        self.force_positive_scale = force_positive_scale

        self.bias = nn.Parameter(torch.full((channels, 1, 1), float(init_bias)))

        if force_positive_scale:
            # raw -> positive via softplus so scale stays > 0
            self.raw_scale = nn.Parameter(torch.full((channels, 1, 1), float(init_scale)))
        else:
            self.scale = nn.Parameter(torch.full((channels, 1, 1), float(init_scale)))

    def forward(self, x):

        if self.force_positive_scale:
            scale = F.softplus(self.raw_scale)
        else:
            scale = self.scale

        return x * scale + self.bias

class ConvertorNoPool(nn.Module):
    def __init__(self, config, force_positive_scale=True):
        super().__init__()
        self.ChannelAffine = PerChannelAffine(config["converter"]["in_channels"], force_positive_scale)
        self.model = nn.Sequential(PerChannelAffine(config["converter"]["in_channels"]), nn.PReLU())

    def forward(self, x):
        return self.model(x)