import datetime
import torch
from tqdm import tqdm
from models.model import RewardPredictionModel
from utils.helpers import get_log_writer
from utils.helpers import device
from torch import nn, optim


def train():
    log_writer = get_log_writer()
    lr = 1e-5
    # 初始化模型、损失函数和优化器
    model = RewardPredictionModel().to(device)
    criterion = nn.HuberLoss()  # 可以使用BCELoss，如果目标是分类
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练循环
    epochs = 30
    train_dataloader, test_dataloader = data_loader()
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for image_feature, mask_feature, iou, image_id, mask_id, image, mask \
                in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', ncols=100):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 将数据传递给模型
                reward_pred = model(image.to(device), mask.to(device))
                # 假设你有一个真实的 IoU 作为目标
                reward_target = iou.float().to(device)  # 这里使用 IoU 作为 reward
                # 计算损失
                loss = criterion(reward_pred, reward_target)
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_steps += 1
        log_writer.add_scalar('Loss/train', train_loss / train_steps, epoch)
        model.eval()
        test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for image_feature, mask_feature, iou, image_id, mask_id, image, mask in test_dataloader:
                # 将数据传递给模型
                reward_pred = model(image.to(device), mask.to(device))
                # 假设你有一个真实的 IoU 作为目标
                reward_target = iou.float().to(device)  # 这里使用 IoU 作为 reward
                # 计算损失
                loss = criterion(reward_pred, reward_target)
                test_loss += loss.item()
                test_steps += 1
        log_writer.add_scalar('Loss/test', test_loss / test_steps, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}, Test Loss: {test_loss / test_steps}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'./checkpoint/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
    latest_model_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    torch.save(model.state_dict(), f'./checkpoint/{latest_model_name}')
    with open('./checkpoint/latest', 'w') as f:
        f.write(latest_model_name)
    test(model, log_writer, test_dataloader, 'Reward')  # 查看测试集分布
    test(model, log_writer, train_dataloader, 'Reward-Train')  # 查看训练集分布
    log_writer.add_text('Model/lr', f'{lr}')
    log_writer.add_text('Model/criterion', f'{criterion}')
    log_writer.add_text(f'Model/model', f'{model}')
    dummy_input = (torch.randn(1, 256, 64, 64).to(device), torch.randn(1, 256, 64, 64).to(device))
    log_writer.add_graph(model, input_to_model=dummy_input)
    log_writer.close()
