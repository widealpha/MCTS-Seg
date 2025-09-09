import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePolicyNetwork(nn.Module):
    """
    仅使用卷积的简单策略网络。
    输入: 图像 [B, C, H, W]
    输出: 每个像素点的概率分布 [B, H, W]
    """
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.conv3(x).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob
