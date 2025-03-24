import torch
import torch.amp
from tqdm import tqdm
from datetime import datetime
from models.model import RewardPredictionModel
from data.loader import get_data_loader
from utils.helpers import get_log_writer, device, setup_seed, get_checkpoints_path, dataset
from torch import nn, optim
import os

checkpoints_path = get_checkpoints_path()

setup_seed()


def train(old_check_point=None):
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
    epochs = 20
    scaler = torch.amp.GradScaler(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            reward = batch['reward'].float().unsqueeze(1).to(device)

            with torch.amp.autocast(device):
                # 将数据传递给模型
                reward_pred = model(image, mask)
                # 计算损失
                loss = criterion(reward_pred, reward)
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_steps += 1
        log_writer.add_scalar('Loss/train', train_loss / train_steps, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}")
        # 评估模型在测试集上的表现
        test(model, test_dataloader, log_writer, epoch, criterion)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(
                checkpoints_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))

    test(model, test_dataloader, log_writer, epoch, criterion)
    latest_model_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    torch.save(model.state_dict(), os.path.join(
        checkpoints_path, latest_model_name))
    torch.save(model.state_dict(), os.path.join(
        checkpoints_path, 'latest.pth'))
    log_writer.add_text('Model/lr', f'{lr}')
    log_writer.add_text('Model/weight_decay', f'{weight_decay}')
    log_writer.add_text('Model/criterion', f'{criterion}')
    log_writer.add_text(f'Model/model', f'{model}')
    dummy_input = (torch.randn(1, 3, sample_width, sample_height).to(device),
                   torch.randn(1, 1, sample_width, sample_height).to(device))
    log_writer.add_graph(model, input_to_model=dummy_input)
    log_writer.close()


def test(model, test_dataloader, log_writer, epoch, criterion):
    # 评估模型在测试集上的表现
    test_loss = 0.0
    test_steps = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f'Test {epoch + 1}'):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            reward = batch['reward'].float().unsqueeze(1).to(device)

            reward_pred = model(image, mask)
            loss = criterion(reward_pred, reward)
            test_loss += loss.item()
            test_steps += 1
    log_writer.add_scalar('Loss/test', test_loss / test_steps, epoch)
    print(f"Epoch [{epoch + 1}], Test Loss: {test_loss / test_steps}")


if __name__ == '__main__':
    old_check_point = os.path.join(
        checkpoints_path, 'latest.pth')
    train(old_check_point=None)
