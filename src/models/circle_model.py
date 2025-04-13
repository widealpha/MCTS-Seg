from torch import nn
import torch


class RewardPredictionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()
        pass

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
