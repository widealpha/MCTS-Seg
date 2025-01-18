import torch
from torch import nn
import torch.nn.functional as F

from unet_model import UNet

class RewardPredictionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNet(n_channels=4, n_classes=1)
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(2**20, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, image, mask):
        # 将image和mask拼接成 (c+1, w, h)
        x = torch.cat((image, mask), dim=1)
        # 通过UNet网络
        x = self.unet(x)
        # 通过卷积层
        x = self.conv(x)
        x = self.conv1(x)
        # 将特征图展平为一个向量
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        return x
