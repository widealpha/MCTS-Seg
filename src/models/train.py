import torch
import torch.amp
from tqdm import tqdm
from datetime import datetime
from models.model import RewardPredictionModel
from data.loader import get_data_loader
from utils.helpers import get_log_writer, get_root_path, device, setup_seed
from torch import nn, optim
import os

root_path = get_root_path()
setup_seed()

def train():
    log_writer = get_log_writer()
    lr = 1e-3
    # 初始化模型、损失函数和优化器
    model = RewardPredictionModel().to(device)
    criterion = nn.MSELoss()  # 可以使用BCELoss，如果目标是分类
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练循环
    epochs = 30
    train_dataloader, test_dataloader = get_data_loader()
    scaler = torch.amp.GradScaler(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            reward = batch['reward'].float().to(device)

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
        # 评估模型在测试集上的表现
        test_loss = 0.0
        test_steps = 0
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                reward = batch['reward'].float().to(device)

                reward_pred = model(image, mask)
                loss = criterion(reward_pred, reward)
                test_loss += loss.item()
                test_steps += 1
        log_writer.add_scalar('Loss/test', test_loss / test_steps, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}, Test Loss: {test_loss / test_steps}")

        torch.save(model.state_dict(), os.path.join(
                root_path, 'results/models', f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))
    latest_model_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    torch.save(model.state_dict(), os.path.join(
            root_path, 'results/models', latest_model_name))
    # test(model, log_writer, test_dataloader, 'Reward')  # 查看测试集分布
    # test(model, log_writer, train_dataloader, 'Reward-Train')  # 查看训练集分布
    log_writer.add_text('Model/lr', f'{lr}')
    log_writer.add_text('Model/criterion', f'{criterion}')
    log_writer.add_text(f'Model/model', f'{model}')
    dummy_input = (torch.randn(1, 256, 64, 64).to(device),
                   torch.randn(1, 256, 64, 64).to(device))
    log_writer.add_graph(model, input_to_model=dummy_input)
    log_writer.close()


if __name__ == '__main__':
    train()
