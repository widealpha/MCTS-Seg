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
from src.utils.helpers import get_baseline_log_path, get_baseline_result_path, get_log_writer, get_test_result_path, load_sam_adapter, setup_seed, calculate_dice, calculate_iou

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


def gengerate_prompt(masks: torch.tensor, point_type, point_num: int = 1, device='cuda'):
    batch_coords = []
    batch_labels = []
    batch_boxes = []
    batch_size = masks.shape[0]

    for b in range(batch_size):
        mask = masks[b].squeeze(0)
        gt = mask.cpu().numpy()
        if point_type == 'foreground':
            y_indices, x_indices = np.where(gt > 0)
        elif point_type == 'background':
            y_indices, x_indices = np.where(gt == 0)
        else:
            raise ValueError('point_type must be foreground or background')
        if len(y_indices) > 0 and len(x_indices) > 0:
            if point_num > len(y_indices):
                chosen = np.random.choice(len(y_indices), point_num, replace=True)
            else:
                chosen = np.random.choice(len(y_indices), point_num, replace=False)
            point_coords = np.array([[x_indices[i], y_indices[i]] for i in chosen])
            batch_coords.append(point_coords)
        else:
            batch_coords.append(None)
    return batch_coords
    # batch_coords = torch.tensor(batch_coords)  # Shape: (B, N, 2)
    # batch_labels = torch.tensor(batch_labels)  # Shape: (B, N)
    # batch_boxes = torch.tensor(batch_boxes)
    # return batch_coords.to(device), batch_labels.to(device), batch_boxes.to(device)


def evaluate_model(model, output_dir, point_type, point_num):
    sam_predictor = get_predictor(model)
    _, test_dataloader = get_baseline_dataloader(batch_size=1, test_batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 4种情况
    cases = [
        ("foreground", 1),
        ("foreground", 0),
        ("background", 1),
        ("background", 0)
    ]
    all_results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_id = batch['image_id'][0] if 'image_id' in batch else 'unknown'
            sam_predictor.set_torch_image(images)
            for point_type, label in cases:
                iou_arr = []
                dice_arr = []
                for repeat in range(100):
                    coords_list = gengerate_prompt(masks, point_type=point_type, point_num=point_num, device=device)
                    # coords_list: list of np.array or None, batch=1
                    if coords_list[0] is None:
                        continue
                    coords = torch.tensor(coords_list[0]).unsqueeze(0).to(device)
                    labels = torch.full((1, coords.shape[1]), label, dtype=torch.int64, device=device)
                    pred, *_ = sam_predictor.predict_torch(coords, labels, None)
                    mask = pred[0].cpu().numpy().squeeze()
                    gt = masks[0].cpu().numpy().squeeze()
                    iou_score = calculate_iou(mask, gt)
                    dice_score = calculate_dice(mask, gt)
                    iou_arr.append(iou_score)
                    dice_arr.append(dice_score)
                # 保存每种case的结果
                case_result = {
                    'image_id': image_id,
                    'point_type': point_type,
                    'label': label,
                    'iou_mean': float(np.mean(iou_arr)) if iou_arr else None,
                    'iou_std': float(np.std(iou_arr)) if iou_arr else None,
                    'dice_mean': float(np.mean(dice_arr)) if dice_arr else None,
                    'dice_std': float(np.std(dice_arr)) if dice_arr else None,
                    'iou_list': iou_arr,
                    'dice_list': dice_arr
                }
                all_results.append(case_result)
                print(f"Image {image_id} | {point_type} label={label} | IOU: {case_result['iou_mean']:.4f}±{case_result['iou_std']:.4f} | DICE: {case_result['dice_mean']:.4f}±{case_result['dice_std']:.4f}")
    # 保存所有结果
    import json
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    # 汇总统计
    summary = {}
    for case in cases:
        pt, lb = case
        ious = [r['iou_mean'] for r in all_results if r['point_type']==pt and r['label']==lb and r['iou_mean'] is not None]
        dices = [r['dice_mean'] for r in all_results if r['point_type']==pt and r['label']==lb and r['dice_mean'] is not None]
        summary[f'{pt}_label{lb}'] = {
            'iou_mean': float(np.mean(ious)) if ious else None,
            'iou_std': float(np.std(ious)) if ious else None,
            'dice_mean': float(np.mean(dices)) if dices else None,
            'dice_std': float(np.std(dices)) if dices else None
        }
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: IOU {v['iou_mean']:.4f}±{v['iou_std']:.4f}, DICE {v['dice_mean']:.4f}±{v['dice_std']:.4f}\n")
    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: IOU {v['iou_mean']:.4f}±{v['iou_std']:.4f}, DICE {v['dice_mean']:.4f}±{v['dice_std']:.4f}")


def test_msa_baseline():
    model = load_sam_adapter()
    return model


if __name__ == "__main__":
    setup_seed()
    model = test_msa_baseline()
    args = parse_args()
    point_num = args.point_num
    point_type = args.point_type
    evaluate_model(model=model,
                   output_dir=os.path.join(
                       get_test_result_path(), f'{model_name}', f'{point_type}_{point_num}'),
                   point_num=point_num,
                   point_type=point_type)
