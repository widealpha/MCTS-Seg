
import torch
from torch import nn, optim
from models.model import RewardPredictionModel
from utils.helpers import get_log_writer, device, dataset


def generate_data():
    pass

def main(check_point=None):
    old_check_point = check_point
    print(f"Start Traning Dataset:{dataset} ...")
    log_writer = get_log_writer()
    train_dataloader, test_dataloader = get_data_loader()
    first_sample = train_dataloader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    lr = 1e-5
    weight_decay = 1e-4
    # 初始化模型、损失函数和优化器
    model = RewardPredictionModel(
        sample_width=sample_width, sample_height=sample_height).to(device)
    if old_check_point:
        model.load_state_dict(torch.load(old_check_point))
        print(f"Loaded model from {old_check_point}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    # 训练循环
    epochs = 60
    scaler = torch.amp.GradScaler(device)
    


if __name__ == 'main':
    pass