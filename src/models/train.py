import torch
import torch.amp
from tqdm import tqdm
from datetime import datetime
from models.model import RewardPredictionModel
from data.loader import get_data_loader
from utils.helpers import get_log_writer, device, setup_seed, get_checkpoints_path
from torch import nn, optim
import os

checkpoints_path = get_checkpoints_path()

setup_seed()


def train(old_check_point=None):
    log_writer = get_log_writer()
    lr = 1e-4
    # 初始化模型、损失函数和优化器
    model = RewardPredictionModel().to(device)
    if old_check_point:
        model.load_state_dict(torch.load(old_check_point))
        print(f"Loaded model from {old_check_point}")
    criterion = nn.MSELoss()  # 修改为分类损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练循环
    epochs = 50
    train_dataloader, test_dataloader = get_data_loader(
        batch_size=2, test_batch_size=4)
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
        if epoch % 5 == 0:
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
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}, Test Loss: {test_loss / test_steps}")
            torch.save(model.state_dict(), os.path.join(
                checkpoints_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))
        else:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}")
    latest_model_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    torch.save(model.state_dict(), os.path.join(
        checkpoints_path, latest_model_name))
    log_writer.add_text('Model/lr', f'{lr}')
    log_writer.add_text('Model/criterion', f'{criterion}')
    log_writer.add_text(f'Model/model', f'{model}')
    dummy_input = (torch.randn(1, 3, 512, 512).to(device),
                   torch.randn(1, 1, 512, 512).to(device))
    log_writer.add_graph(model, input_to_model=dummy_input)
    log_writer.close()


if __name__ == '__main__':
    old_check_point = os.path.join(
        checkpoints_path, '2025-01-18_15-31-04.pth')
    train(old_check_point=None)
