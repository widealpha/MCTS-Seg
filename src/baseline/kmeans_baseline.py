import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import scipy.ndimage as ndi

import torch.nn as nn
import torch.optim as optim

from src.baseline.MedSAM.segment_anything.predictor import SamPredictor
from src.data.baseline_dataloader import get_baseline_dataloader

from src.cfg import parse_args

from src.utils.helpers import get_baseline_result_path, get_log_writer, load_sam, setup_seed, calculate_dice, calculate_iou

model_name = 'kmeans'


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
    # 不再使用sam_predictor
    _, test_dataloader = get_baseline_dataloader(
        batch_size=1, test_batch_size=1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iou = []
    dice = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        for image, gt, image_id in zip(batch['image'], batch['mask'], batch['image_id']):
            # image: (C, H, W)
            img_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            gt = gt[0].cpu().numpy()  # (H, W)
            h, w, c = img_np.shape
            flat_img = img_np.reshape(-1, c)

            # KMeans 聚类
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = kmeans.fit_predict(flat_img)
            labels_img = labels.reshape(h, w)

            # 遍历每个聚类标签，计算 IOU，选最大
            best_iou = 0
            best_mask = None
            best_dice = 0
            for label in range(n_clusters):
                mask = (labels_img == label).astype(np.uint8)
                cur_iou = calculate_iou(mask, gt)
                cur_dice = calculate_dice(mask, gt)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_mask = mask
                    best_dice = cur_dice
            # 提取mask中的最大联通分量
            labeled_mask, num_features = ndi.label(best_mask)
            if num_features > 0:
                sizes = ndi.sum(best_mask, labeled_mask, range(1, num_features + 1))
                largest_cc = (labeled_mask == (np.argmax(sizes) + 1)).astype(np.uint8)
                best_iou = calculate_iou(largest_cc, gt)
                best_dice = calculate_dice(largest_cc, gt)
                best_mask = largest_cc
            iou.append(best_iou)
            dice.append(best_dice)
            mask_to_save = (best_mask * 255).astype(np.uint8)
            Image.fromarray(mask_to_save).save(
                os.path.join(output_dir, f"{image_id}.png"), format='PNG')
            with open(os.path.join(output_dir, f"{image_id}.txt"), "w") as f:
                f.write(f"Best IOU: {best_iou:.4f}\n")
                f.write(f"Best DICE: {best_dice:.4f}\n")

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
    # KMeans 不需要模型，返回 None
    return None


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
