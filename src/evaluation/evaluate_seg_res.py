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
from src.utils.helpers import get_baseline_log_path, get_baseline_result_path, get_log_writer, load_sam, load_sam_adapter, setup_seed, calculate_dice, calculate_iou

model_name = 'test'
pattern = re.compile(r'Path:\s*(\[\[.*?\]\])')


def get_sam_predictor():
    """
    获取 SAM 预测器
    """
    args = parse_args()
    if args.bone == 'sam':
        sam = load_sam()
        sam_predictor = SamPredictor(sam)
    elif args.bone == 'msa':
        sam = load_sam_adapter()
        sam_predictor = MSAPredictor(sam)
    return sam_predictor


def gengerate_prompt(image_ids, prompt_dir, device='cuda'):
    batch_coords = []
    batch_labels = []
    batch_size = len(image_ids)

    for b in range(batch_size):
        with open(os.path.join(prompt_dir, f"{image_ids[b]}_result.txt"), 'r') as f:
            line = f.readline()
            match = pattern.search(line)
            if match:
                path_str = match.group(1)
                path_list = ast.literal_eval(path_str)  # 安全地转成list
                # print(path_list)
                # 取最后一个点
                prompts = [*path_list[-1], *path_list[-2]]
                # prompts = path_list[-1]
                batch_coords.append(prompts)
                batch_labels.append([1 for _ in range(len(prompts))])
            else:
                print("Path not found")
    batch_coords = torch.tensor(batch_coords)  # Shape: (B, N, 2)
    batch_labels = torch.tensor(batch_labels)  # Shape: (B, N)
    return batch_coords.to(device), batch_labels.to(device)


def evaluate_model(output_dir, prompt_dir):
    sam_predictor = get_sam_predictor()
    _, test_dataloader = get_baseline_dataloader(
        batch_size=1, test_batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iou = []
    dice = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_ids = batch['image_id']
            image_array = images[0].permute(1, 2, 0).cpu().numpy()
            sam_predictor.set_image(image_array)
            coords, labels = gengerate_prompt(
                    image_ids, prompt_dir, device=device)
            pred, *_ = sam_predictor.predict_torch(coords, labels, multimask_output=False)
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
        var_iou = np.var(iou)
        mean_dice = np.mean(dice)
        std_dice = np.std(dice)
        var_dice = np.var(dice)
        print(f"Mean IOU: {mean_iou:.4f} ± {std_iou:.4f} ± {var_iou:.4f}")
        print(f"Mean DICE: {mean_dice:.4f} ± {std_dice:.4f} ± {var_dice:.4f}")

        f.write(f"Mean IOU: {mean_iou}, MSD: {std_iou}\n")
        f.write(f"Mean DICE: {mean_dice}, MSD: {std_dice}\n")


if __name__ == "__main__":
    setup_seed()
    args = parse_args()
    point_num = args.point_num
    point_type = args.point_type
    evaluate_model(output_dir=os.path.join(
        get_baseline_result_path(), f'{model_name}', f'{point_type}_{point_num}'),
        # prompt_dir=os.path.join('/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2018_2025-04-28_19-34-55')
        prompt_dir=os.path.join('/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2016_2025-04-26_15-16-05')
    )
