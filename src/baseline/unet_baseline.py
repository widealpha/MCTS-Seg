import os
import numpy as np
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from src.data.baseline_dataloader import get_baseline_dataloader

from src.cfg import parse_args

from src.models.unet_model import UNet
from src.utils.helpers import get_baseline_log_path, get_log_writer, setup_seed, calculate_dice, calculate_iou


class UNetBaselineModel(nn.Module):
    def __init__(self):
        super(UNetBaselineModel, self).__init__()
        self.unet = UNet(n_channels=3, n_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.unet(x)
        x = self.sigmoid(x)
        return x


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Testing function


def test_model(model, dataloader, device):
    model.eval()
    count = 0
    dices_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images).squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            masks = masks.squeeze(1)
            for mask, gt in zip(outputs, masks):
                dices_scores.append(calculate_dice(mask, gt))
                iou_scores.append(calculate_iou(mask, gt))

    return np.mean(iou_scores), np.mean(dices_scores), np.std(iou_scores), np.std(dices_scores)


def test_unet_baseline():
    dataset = parse_args().dataset
    # Hyperparameters
    epochs = 100
    lr = 1e-5
    weight_decay = 1e-4
    patience = 10  # 早停的耐心值
    best_iou = 0.0
    patience_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Training Dataset:{dataset} ...")
    log_path = os.path.join(get_baseline_log_path(), 'unet')
    # 获取日志路径并初始化 TensorBoard
    writer = get_log_writer(log_path)

    train_dataloader, test_dataloader = get_baseline_dataloader(
        batch_size=8, test_batch_size=8)
    # Model, loss, optimizer
    model = UNetBaselineModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Training and testing loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(
            model, train_dataloader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        iou, dice, iou_std, dice_std = test_model(model, test_dataloader, device)
        print(f"IoU Score: {iou:.4f} ± {iou_std:.4f}")
        print(f"Dice Score: {dice:.4f} ± {dice_std:.4f}")

        # 写入 TensorBoard
        writer.add_scalar("Train/Loss", train_loss, epoch + 1)
        writer.add_scalar("Test/IoU", iou, epoch + 1)
        writer.add_scalar("Test/IoU_std", iou_std, epoch + 1)
        writer.add_scalar("Test/Dice", dice, epoch + 1)
        writer.add_scalar("Test/Dice_std", dice_std, epoch + 1)

        # 早停机制
        if iou > best_iou:
            best_iou = iou
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), os.path.join(
                log_path, 'unet_baseline_best.pth'))
            print("Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Save the final model
    torch.save(model.state_dict(), os.path.join(
        log_path, 'unet_baseline_final.pth'))
    print("Final model saved!")

    # 关闭 TensorBoard
    writer.close()


if __name__ == "__main__":
    setup_seed()
    test_unet_baseline()
