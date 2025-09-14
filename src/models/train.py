import torch
import torch.amp
from torch.nn import MarginRankingLoss
from tqdm import tqdm
from datetime import datetime
from src.models.model import RewardPredictionModel, SimpleRewardModel, UNetRewardModel
from src.data.loader import get_data_loader
from src.utils.helpers import get_log_writer, setup_seed, get_checkpoints_path
from src.cfg import parse_args
from torch import nn, optim
import os

device = parse_args().device
dataset = parse_args().dataset
checkpoints_path = os.path.join(get_checkpoints_path(), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(checkpoints_path, exist_ok=True)

def train(old_check_point=None):
    print(f"Start Traning Dataset:{dataset} ...")
    
    train_dataloader, test_dataloader = get_data_loader(batch_size=16)
    first_sample = train_dataloader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    lr = 1e-5
    weight_decay = 1e-1
    # 初始化模型、损失函数和优化器
    model = RewardPredictionModel().to(device)
    if old_check_point:
        model.load_state_dict(torch.load(old_check_point))
        print(f"Loaded model from {old_check_point}")
    criterion = MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    # 训练循环
    epochs = 100
    patience = 100  # 早停的耐心值
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
            mask1 = batch['mask1'].to(device)
            mask2 = batch['mask2'].to(device)
            reward1 = batch['reward1'].float().unsqueeze(1).to(device)
            reward2 = batch['reward2'].float().unsqueeze(1).to(device)

            with torch.amp.autocast(device):
                reward_pred1 = model(image, mask1)
                reward_pred2 = model(image, mask2)
                # 计算标签：reward1 > reward2 -> 1，否则-1
                target = torch.sign(reward1 - reward2)
                target[target == 0] = 1  # 避免0标签
                loss = criterion(reward_pred1, reward_pred2, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_steps += 1
        log_writer.add_scalar('Train/Loss', train_loss / train_steps, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss:{train_loss / train_steps}")
        
        # 评估模型在测试集上的表现
        loss, *rest = test(model, test_dataloader, log_writer, epoch)
        if loss < test_loss:
            test_loss = loss
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), os.path.join(
                checkpoints_path, f'best_model_{epoch}.pth'))
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
    total_loss = 0.0
    total_samples = 0
    criterion = MarginRankingLoss(margin=1.0)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            image = batch['image'].to(device)
            mask1 = batch['mask1'].to(device)
            mask2 = batch['mask2'].to(device)
            reward1 = batch['reward1'].float().unsqueeze(1).to(device)
            reward2 = batch['reward2'].float().unsqueeze(1).to(device)

            reward_pred1 = model(image, mask1)
            reward_pred2 = model(image, mask2)
            target = torch.sign(reward1 - reward2)
            target[target == 0] = 1
            loss = criterion(reward_pred1, reward_pred2, target)
            total_loss += loss.item() * image.size(0)
            total_samples += image.size(0)

    avg_loss = total_loss / total_samples
    if log_writer is not None:
        log_writer.add_scalar('Test/PairwiseLoss', avg_loss, epoch)

    print(f"Epoch [{epoch + 1}], Pairwise Ranking Loss: {avg_loss:.4f}")
    return avg_loss, None, None


if __name__ == '__main__':
    setup_seed()
    old_check_point = os.path.join(checkpoints_path, 'latest.pth')
    train(old_check_point=None)
