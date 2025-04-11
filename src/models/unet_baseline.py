import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from models.unet_model import UNet
from data.loader import get_data_loader
from utils.helpers import get_log_writer, dataset, get_log_writer, setup_seed

setup_seed()


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.unet = UNet(n_channels=3, n_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.unet(x)
        x = self.sigmoid(x)
        return x

# Training function


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
    dice_score = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            outputs = (outputs > 0.5).float()

            # 计算交集和并集
            intersection = (outputs * masks).sum()
            union = outputs.sum() + masks.sum() - intersection  # 修复 IoU 的并集计算
            dice_denominator = outputs.sum() + masks.sum()  # 修复 Dice 的分母计算

            # 避免分母为 0 的情况
            if union > 0:
                iou_score += (intersection / union).item()
            if dice_denominator > 0:
                dice_score += (2.0 * intersection / dice_denominator).item()

    return dice_score / len(dataloader), iou_score / len(dataloader)


# Main script
if __name__ == "__main__":
    # Hyperparameters
    epochs = 50
    lr = 1e-5
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Traning Dataset:{dataset} ...")

    # 获取日志路径并初始化 TensorBoard
    writer = get_log_writer()

    train_dataloader, test_dataloader = get_data_loader(
        batch_size=8, test_batch_size=8, per_image_mask=1)
    # Model, loss, optimizer
    model = BaselineModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Training and testing loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(
            model, train_dataloader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        dice, iou = test_model(model, test_dataloader, device)
        print(f"IoU Score: {iou:.4f}")
        print(f"Dice Score: {dice:.4f}")

        # 写入 TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Dice/Test", dice, epoch + 1)
        writer.add_scalar("IoU/Test", iou, epoch + 1)

    # Save the model
    os.makedirs("result/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "result/checkpoints/unet_baseline.pth")
    print("Model saved!")

    # 关闭 TensorBoard
    writer.close()
