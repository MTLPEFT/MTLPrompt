
import torch.nn as nn
from timm.models.layers import trunc_normal_
BatchNorm2d = nn.BatchNorm2d




class ConvHead3x3(nn.Module):
    def __init__(self, cfg, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))