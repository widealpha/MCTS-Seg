import torch
from pytorch_model_summary import summary

from src.models.model import RewardPredictionModel, SimpleRewardModel,UNetRewardModel

# Model
print('==> Building model..')
model = RewardPredictionModel()

dummy_input1 = torch.randn(1, 3, 512, 512)
dummy_input2 = torch.randn(1, 1, 512, 512)

print(summary(model,dummy_input1,dummy_input2, show_input=False, show_hierarchical=False, max_depth=2))
