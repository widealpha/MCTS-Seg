import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torch.optim as optim

from src.baseline.MedSAM.segment_anything.predictor import SamPredictor
from src.data.baseline_dataloader import get_baseline_dataloader

from src.cfg import parse_args

from src.utils.helpers import get_baseline_result_path, get_log_writer, load_sam, setup_seed, calculate_dice, calculate_iou

model_name = 'sam'


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


def evaluate_model(model, output_dir, point_type, point_num):
    sam_predictor = SamPredictor(model)
    _, test_dataloader = get_baseline_dataloader(
        batch_size=1, test_batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iou = []
    dice = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            for image, gt, image_id in zip(batch['image'], batch['mask'], batch['image_id']):
                image = image.permute(1, 2, 0).cpu().numpy()
                gt = gt[0].cpu().numpy()
                sam_predictor.set_image(image)
                point_coords = None
                point_labels = None
                box = None
                if point_type == 'none':
                    point_coords = None
                    point_labels = None
                elif point_type == 'centroid':
                    # Calculate the center point of the ground truth mask
                    y_indices, x_indices = np.where(gt > 0)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))
                        point_coords = np.array([[center_x, center_y]])
                        point_labels = np.array([1])
                    else:
                        point_coords = None
                        point_labels = None
                elif point_type == 'random':
                    # Randomly sample points from the ground truth mask
                    y_indices, x_indices = np.where(gt > 0)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        random_indices = np.random.choice(
                            len(y_indices), point_num, replace=False)
                        point_coords = np.array(
                            [[x_indices[i], y_indices[i]] for i in random_indices])
                        point_labels = np.array([1] * point_num)
                    else:
                        point_coords = None
                        point_labels = None
                elif point_type == 'center':
                    # Calculate the center point of the ground truth mask
                    y_indices, x_indices = np.where(gt > 0)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        center_y = int(
                            (np.min(y_indices) + np.max(y_indices)) / 2)
                        center_x = int(
                            (np.min(x_indices) + np.max(x_indices)) / 2)
                        point_coords = np.array([[center_x, center_y]])
                        point_labels = np.array([1])
                    else:
                        point_coords = None
                        point_labels = None
                elif point_type == 'box':
                    # Calculate the bounding box of the ground truth mask
                    y_indices, x_indices = np.where(gt > 0)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        min_y, max_y = np.min(y_indices), np.max(y_indices)
                        min_x, max_x = np.min(x_indices), np.max(x_indices)
                        box = np.array([min_x, min_y, max_x, max_y])
                    else:
                        point_coords = None
                        point_labels = None
                # 预测掩码
                masks, *_ = sam_predictor.predict(
                    point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=False)
                mask = masks[0]
                iou.append(calculate_iou(mask, gt))
                dice.append(calculate_dice(mask, gt))
                mask = mask > 0.5
                mask = mask * 255
                mask = mask.astype(np.uint8)
                Image.fromarray(mask).save(
                    os.path.join(output_dir, f"{image_id}.png"), format='PNG')
                with open(os.path.join(output_dir, f"{image_id}.txt"), "w") as f:
                    if point_coords is not None and point_labels is not None:
                        f.write(f"Coordinates: {point_coords.tolist()}\n")
                        f.write(f"Labels: {point_labels.tolist()}\n")
                    if box is not None:
                        f.write(f"Bounding Box: {box.tolist()}\n")
                    f.write(f"IOU: {iou[-1]:.4f}\n")
                    f.write(f"DICE: {dice[-1]:.4f}\n")
                # masks, scores, logits = sam_predictor.predict(
                #     point_coords=point_coords, point_labels=point_labels, multimask_output=False)
                # mask = masks[0]
                # iou.append(calculate_iou(mask, gt))
                # dice.append(calculate_dice(mask, gt))
                # mask = mask > 0.5
                # mask = mask * 255
                # mask = mask.astype(np.uint8)
                # # 保存为 PNG 图像
                # Image.fromarray(mask).save(
                #     os.path.join(output_dir, f"{image_id}.png"), format='PNG')
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
    model = load_sam()
    return model


if __name__ == "__main__":
    setup_seed()
    model = test_msa_baseline()
    args = parse_args()
    point_num = args.point_num
    point_type = args.point_type
    evaluate_model(model=model,
                   output_dir=os.path.join(
                       get_baseline_result_path(), f'{model_name}', f'{point_type}_{point_num}'),
                   point_num=point_num,
                   point_type=point_type)
