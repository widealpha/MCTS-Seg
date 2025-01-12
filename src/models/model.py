import torch
from torch import nn
import torch.nn.functional as F

from unet_model import UNet


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class RewardPredictionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNet(n_channels=3, n_classes=2)
        # self.mask_unet = UNet(n_channels=1, n_classes=2)
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)  # 输出通道数 128，卷积核 3x3
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # 输出通道数 128，卷积核 3x3
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # 输出通道数 128，卷积核 3x3
        # 定义全连接层
        self.fc1 = nn.Linear(16 * 256 * 256, 16)

        self.fc2 = nn.Linear(16, 1)
        # self.activation = nn.ReLU()

    def forward(self, image, mask):
        image_feature = self.unet(image)
        mask_feature = self.unet(mask)
        x = torch.cat((image_feature, mask_feature), dim=1)  # shape: (4, 1024, 1024)
        # 通过卷积层
        x = self.conv(x)  # shape: (128, 64, 64)
        x = self.conv1(x)  # shape: (128, 64, 64)
        # x = self.conv2(x)  # shape: (128, 64, 64)

        # 将特征图展平为一个向量
        x = x.view(x.size(0), -1)  # 展平，变成 (batch_size, 128 * 64 * 64)
        # 通过全连接层
        x = self.fc1(x)  # shape: (batch_size, 16)
        x = self.fc2(x)  # shape: (batch_size, 1)
        # x = self.activation(x)
        # x = F.sigmoid(x)
        return x
