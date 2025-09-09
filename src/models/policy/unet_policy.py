import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import UNet


class UNetPolicyNetwork(nn.Module):
    """
    使用 UNet 的策略网络。
    输入: 图像 [B, C, H, W]
    输出: 每个像素点的概率分布 [B, H, W]
    """

    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()
        self.unet = UNet(in_channels, out_channels)
        self.conv_out = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x):
        feat = self.unet(x)
        logits = self.conv_out(feat).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob
