import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import fcn_resnet101

from src.data.baseline_dataloader import get_baseline_dataloader

from src.cfg import parse_args

from src.utils.helpers import get_baseline_log_path, get_baseline_result_path, get_device, get_log_writer, setup_seed, calculate_dice, calculate_iou

model_name = 'fcn_resnet101'


class FCNBaseline(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = fcn_resnet101(weights='DEFAULT')
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x['out'])  # 添加 Sigmoid 激活


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


def evaluate_model(model, output_dir):
    _, test_dataloader = get_baseline_dataloader(
        batch_size=8, test_batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_ids = batch['image_id']
            outputs = model(images).squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            masks = masks.squeeze(1)
            for mask, image_id in zip(outputs, image_ids):
                # 将浮点型数据归一化到 [0, 255] 并转换为 uint8
                mask = mask.cpu().numpy() > 0.5
                mask = mask * 255
                mask = mask.astype(np.uint8)
                # 保存为 PNG 图像
                Image.fromarray(mask).save(
                    os.path.join(output_dir, f"{image_id}.png"), format='PNG')


def train_fcn_baseline():
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
    log_path = os.path.join(get_baseline_log_path(), f'model_name')
    # 获取日志路径并初始化 TensorBoard
    writer = get_log_writer(log_path)

    train_dataloader, test_dataloader = get_baseline_dataloader(
        batch_size=8, test_batch_size=8)
    # Model, loss, optimizer
    model = FCNBaseline().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Training and testing loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(
            model, train_dataloader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        iou, dice, iou_std, dice_std = test_model(
            model, test_dataloader, device)
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
                log_path, f'{model_name}_baseline_best.pth'))
            print("Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Save the final model
    torch.save(model.state_dict(), os.path.join(
        log_path, f'{model_name}_baseline_final.pth'))
    print("Final model saved!")

    # 关闭 TensorBoard
    writer.close()
    return model


if __name__ == "__main__":
    setup_seed()
    model = train_fcn_baseline()
    evaluate_model(model=model, output_dir=os.path.join(
        get_baseline_result_path(), f'{model_name}'))
