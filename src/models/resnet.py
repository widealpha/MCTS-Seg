import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class DualInputResNet(nn.Module):
    def __init__(self, block, layers, input_w, input_h, num_classes=1):
        super(DualInputResNet, self).__init__()
        self.input_w = input_w  # 输入宽度
        self.input_h = input_h  # 输入高度
        
        # 分支1: 处理3通道输入 (RGB图像)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 分支2: 处理1通道输入 (如深度图)
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 合并通道后的主网络 (32+32=64通道)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            self.feature_size = self._get_feature_size()
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _get_feature_size(self):
        """计算全连接层前的特征维度"""
        x1 = torch.randn(1, 3, self.input_h, self.input_w)
        x2 = torch.randn(1, 1, self.input_h, self.input_w)
        x = self.forward_features(x1, x2)
        return x.view(-1).shape[0]

    def forward_features(self, x1, x2):
        # 分支处理
        x1 = self.branch1(x1)  # [B,32,H,W]
        x2 = self.branch2(x2)  # [B,32,H,W]
        x = torch.cat([x1, x2], dim=1)  # 通道拼接 [B,64,H,W]
        
        # 主网络
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 使用示例
def dual_resnet18(w, h):
    return DualInputResNet(ResidualBlock, [2,2,2,2], input_w=w, input_h=h)

if __name__ == "__main__":
    # 初始化模型 (输入尺寸512x512)
    model = dual_resnet18(w=512, h=512)
    
    # 测试输入
    x_rgb = torch.randn(2, 3, 512, 512)  # 3通道输入
    x_depth = torch.randn(2, 1, 512, 512) # 1通道输入
    output = model(x_rgb, x_depth)
    print("Output shape:", output.shape)  # 期望输出: torch.Size([2,1])