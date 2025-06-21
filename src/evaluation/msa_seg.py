import ast
import os
import re
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
match = re.compile(r'Path:\s*(\[\[.*?\]\])')

def get_predictor(model):
    sam_predictor = MSAPredictor(model)
    return sam_predictor


def gengerate_prompt(image_ids, prompt_dir, device='cuda'):
    batch_coords = []
    batch_labels = []
    batch_size = len(image_ids)

    for b in range(batch_size):
        with open(os.path.join(prompt_dir, f"{image_ids[b]}_result.txt"), 'r') as f:
            line = f.readline()
            global match
            match = match.search(line)
            if match:
                path_str = match.group(1)
                path_list = ast.literal_eval(path_str)  # 安全地转成list
                # print(path_list)
                batch_coords.append([path_list[-1][-1]])
                batch_labels.append([1])
            else:
                print("Path not found")
        # mask = masks[b].squeeze(0)  # Remove channel dimension if present
        # # Get indices where mask > 0
        # gt = mask.cpu().numpy()
        # y_indices, x_indices = np.where(gt > 0)
        # if point_type == 'center':
        #     if len(y_indices) > 0 and len(x_indices) > 0:
        #         center_y = int((np.min(y_indices) + np.max(y_indices)) / 2)
        #         center_x = int((np.min(x_indices) + np.max(x_indices)) / 2)
        #         point_coords = np.array([[center_x, center_y]])
        #         point_labels = np.array([1])
        #     else:
        #         point_coords = None
        #         point_labels = None
        # elif point_type == 'random':
        #     y_indices, x_indices = np.where(gt > 0)
        #     if len(y_indices) > 0 and len(x_indices) > 0:
        #         random_indices = np.random.choice(
        #             len(y_indices), point_num, replace=False)
        #         point_coords = np.array(
        #             [[x_indices[i], y_indices[i]] for i in random_indices])
        #         point_labels = np.array([1] * point_num)
        #     else:
        #         point_coords = None
        #         point_labels = None
        # elif point_type == 'centroid':
        #     y_indices, x_indices = np.where(gt > 0)
        #     if len(y_indices) > 0 and len(x_indices) > 0:
        #         centroid_y = int(np.mean(y_indices))
        #         centroid_x = int(np.mean(x_indices))
        #         point_coords = np.array([[centroid_x, centroid_y]])
        #         point_labels = np.array([1])
        #     else:
        #         point_coords = None
        #         point_labels = None

        # batch_coords.append(point_coords)
        # batch_labels.append(point_labels)

    batch_coords = torch.tensor(batch_coords)  # Shape: (B, N, 2)
    batch_labels = torch.tensor(batch_labels)  # Shape: (B, N)
    return batch_coords.to(device), batch_labels.to(device)


def evaluate_model(model, output_dir, point_type, point_num):
    sam_predictor = get_predictor(model)
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
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            sam_predictor.set_torch_image(images)
            if point_type == 'none':
                coords = None
                labels = None
            else:
                coords, labels = gengerate_prompt(
                    masks, point_type=point_type, point_num=point_num, device=device)
            pred, *_ = sam_predictor.predict_torch(coords, labels)
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
    args = parse_args()
    point_num = args.point_num
    point_type = args.point_type
    evaluate_model(model=model,
                   output_dir=os.path.join(
                       get_baseline_result_path(), f'{model_name}', f'{point_type}_{point_num}'),
                   point_num=point_num,
                   point_type=point_type)
