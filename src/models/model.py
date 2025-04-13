import torch
from torch import nn
import torch.nn.functional as F


import torchvision
from .resnet import DualInputResNet, ResidualBlock
from .unet_model import UNet


class RewardPredictionModel(nn.Module):
    def __init__(self, sample_width=260, sample_height=260, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.resnet = DualInputResNet(
        #     ResidualBlock, [2, 2, 2, 2], sample_width, sample_height)
        self.encoder = torchvision.models.resnet152(weights='DEFAULT')
        self.encoder.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(1000, 1)

        self.unet = UNet(n_channels=4, n_classes=1)
        # 定义卷积层和池化层
        self.conv = nn.Conv2d(in_channels=1, out_channels=8,
                              kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(
            self._get_conv_output_size((4, sample_width, sample_height)), 16)
        self.fc2 = nn.Linear(16, 1)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.unet(input)
            output = self.conv(output)
            output = self.pool(output)
            output = self.conv1(output)
            output = self.pool(output)
            return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)  # 拼接成2通道
        x = self.encoder(x)
        return self.fc(x) # 输出范围[0,1]
        # # 将image和mask拼接成 (B, C+1, H, W)
        # x = torch.cat((image, mask), dim=1)
        # # 通过UNet网络
        # x = self.unet(x)  # (B, 1, H, W)
        # # 通过卷积层和池化层
        # x = self.conv(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # # 将特征图展平为一个向量
        # x = x.view(x.size(0), -1)  # (B, 1*H*W)
        # # 通过全连接层
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # return x
