import torch
import torch.amp
from tqdm import tqdm
from datetime import datetime
from src.models.model import RewardPredictionModel, SimpleRewardModel
from src.data.loader import get_data_loader
from src.utils.helpers import get_log_writer, setup_seed, get_checkpoints_path
from src.cfg import parse_args
from torch import nn, optim
import os

device = parse_args().device
dataset = parse_args().dataset
checkpoints_path = os.path.join(get_checkpoints_path(), 'simple')

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path, exist_ok=True)


def train(old_check_point=None):
    print(f"Start Traning Dataset:{dataset} ...")
    
    train_dataloader, test_dataloader = get_data_loader()
    first_sample = train_dataloader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    lr = 1e-3
    weight_decay = 1e-4
    # 初始化模型、损失函数和优化器
    model = SimpleRewardModel().to(device)
    if old_check_point:
        model.load_state_dict(torch.load(old_check_point))
        print(f"Loaded model from {old_check_point}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    # 训练循环
    epochs = 100
    patience = 15  # 早停的耐心值
    scaler = torch.amp.GradScaler(device)
    test_loss = float('inf')
    patience_counter = 0
    
    log_writer = get_log_writer()
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
        log_writer.add_scalar('Train/Loss', train_loss / train_steps, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}")
        
        # 评估模型在测试集上的表现
        loss, *_ = test(model, test_dataloader, log_writer, epoch)
        if loss < test_loss:
            test_loss = loss
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), os.path.join(
                checkpoints_path, 'best_model.pth'))
            print("Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(
                checkpoints_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))

    test(model, test_dataloader, log_writer, epoch)
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


def test(model, test_dataloader, log_writer, epoch):
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            reward = batch['reward'].float().unsqueeze(1).to(device)

            reward_pred = model(image, mask)

            # 计算MSE和MAE
            batch_mse = torch.mean((reward_pred - reward) ** 2)
            batch_mae = torch.mean(torch.abs(reward_pred - reward))

            total_mse += batch_mse.item() * reward.size(0)
            total_mae += batch_mae.item() * reward.size(0)
            total_samples += reward.size(0)

    final_mse = total_mse / total_samples
    final_rmse = final_mse ** 0.5
    final_mae = total_mae / total_samples

    if log_writer is not None:
        log_writer.add_scalar('Test/MSE', final_mse, epoch)
        log_writer.add_scalar('Test/RMSE', final_rmse, epoch)
        log_writer.add_scalar('Test/MAE', final_mae, epoch)

    print(
        f"Epoch [{epoch + 1}], MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    return final_mse, final_rmse, final_mae


if __name__ == '__main__':
    setup_seed()
    old_check_point = os.path.join(checkpoints_path, 'latest.pth')
    train(old_check_point=None)
