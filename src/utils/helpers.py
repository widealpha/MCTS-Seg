import os
import random
from datetime import datetime

import numpy as np
import torch
from segment_anything import sam_model_registry
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "BraTS"


def get_log_writer():
    return SummaryWriter(os.path.join(get_root_path(), 'results/logs', f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'))


def get_root_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '../../')


def get_data_path():
    return os.path.join(get_root_path(), 'data', dataset)


def setup_seed(seed: int = 2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    np.random.seed(seed)
    random.seed(seed)

    # 保证结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sam():
    path = os.path.join(
        get_root_path(), 'data/external/sam_vit_h_4b8939.pth')
    sam = sam_model_registry["vit_h"](checkpoint=path)
    sam.eval()
    return sam.to(device)


def calculate_iou(pred_mask, true_mask):
    # 假设 pred_mask 和 true_mask 是二值化的掩码（1表示目标区域，0表示背景）
    # 获取前景区域
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5

    # 计算 IoU（Intersection over Union）
    intersection = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()

    iou = intersection / union
    return iou
