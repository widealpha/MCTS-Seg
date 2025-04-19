import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.baseline.MedSAM.segment_anything.predictor import SamPredictor
from src.data.baseline_dataloader import get_baseline_dataloader

from src.cfg import parse_args

from src.models.msa_predictor import MSAPredictor
from src.models.unet_model import UNet
from src.utils.helpers import get_baseline_log_path, get_baseline_result_path, get_log_writer, load_sam_adapter, setup_seed, calculate_dice, calculate_iou

model_name = 'medical_sam_adapter'


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


def get_predictor(model):
    sam_predictor = MSAPredictor(model)
    return sam_predictor


def gengerate_prompt(masks: torch.tensor, num_points: int = 1, device='cuda'):
    point_coords: torch.Tensor = None,
    point_labels: torch.Tensor = None,
    batch_size = masks.shape[0]
    point_coords = []
    point_labels = []
    for b in range(batch_size):
        mask = masks[b].squeeze(0)  # Remove channel dimension if present
        indices = torch.nonzero(mask > 0, as_tuple=False)  # Get indices where mask > 0
        if len(indices) > 0:
            batch_coords = []
            batch_labels = []
            if len(indices) >= num_points:
                selected_indices = indices[torch.randperm(len(indices))[:num_points]]
            else:
                selected_indices = indices[torch.randint(0, len(indices), (num_points,))]
            for point in selected_indices:
                batch_coords.append(point.tolist())
                batch_labels.append(1)
                
            point_coords.append(batch_coords)
            point_labels.append(batch_labels)
            
    point_coords = torch.tensor(point_coords)  # Shape: (B, N, 2)
    point_labels = torch.tensor(point_labels)  # Shape: (B, N)
    return point_coords.to(device), point_labels.to(device)
    


def evaluate_model(model, output_dir):
    sam_predictor = get_predictor(model)
    _, test_dataloader = get_baseline_dataloader(
        batch_size=1, test_batch_size=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iou = []
    dice = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            sam_predictor.set_torch_image(images)
            coords, labels = gengerate_prompt(masks, num_points=10, device=device)
            pred, *_ = sam_predictor.predict_torch()
            for mask, gt, image_id in zip(pred, masks, batch['image_id']):
                mask = mask.cpu().numpy().squeeze()
                gt = gt.cpu().numpy().squeeze()
                iou.append(calculate_iou(mask, gt))
                dice.append(calculate_dice(mask, gt))
                mask = mask > 0.5
                mask = mask * 255
                mask = mask.astype(np.uint8)
                # 保存为 PNG 图像
                Image.fromarray(mask).save(
                    os.path.join(output_dir, f"{image_id}.png"), format='PNG')
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        mean_iou = np.mean(iou)
        std_iou = np.std(iou)
        mean_dice = np.mean(dice)
        std_dice = np.std(dice)
        print(f"Mean IOU: {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"Mean DICE: {mean_dice:.4f} ± {std_dice:.4f}")

        f.write(f"Mean IOU: {mean_iou}, MSD: {std_iou}\n")
        f.write(f"Mean DICE: {mean_dice}, MSD: {std_dice}\n")


def test_msa_baseline():
    model = load_sam_adapter()
    return model


if __name__ == "__main__":
    setup_seed()
    model = test_msa_baseline()
    evaluate_model(model=model, output_dir=os.path.join(
        get_baseline_result_path(), f'{model_name}'))
