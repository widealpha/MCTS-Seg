import os
import random
from datetime import datetime

import numpy as np
import torch
from segment_anything import sam_model_registry
from torch.utils.tensorboard import SummaryWriter
from src.cfg import parse_args


def get_dataset():
    return parse_args().dataset


def get_device():
    return parse_args().device


def get_root_path():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))


def get_data_path():
    res = os.path.join(get_root_path(), 'data', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_checkpoints_path():
    res = os.path.join(get_root_path(), 'checkpoints', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_train_log_path():
    res = os.path.join(get_root_path(), 'logs', 'train', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_baseline_log_path():
    res = os.path.join(get_root_path(), 'logs', 'baseline', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_baseline_result_path():
    res = os.path.join(get_root_path(), 'results', 'baseline', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_mcts_result_path():
    res = os.path.join(get_root_path(), 'results', 'mcts', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_log_writer(log_path=None):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not log_path:
        log_path = get_train_log_path()
    return SummaryWriter(os.path.join(log_path, time_str), time_str)


def setup_seed(seed: int = 2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    np.random.seed(seed)
    random.seed(seed)

    # 保证结果可复现
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_sam():
    return load_sam_vit_h()


def load_sam_vit_h():
    path = os.path.join(
        get_root_path(), 'data/external/sam_vit_h_4b8939.pth')
    sam = sam_model_registry["vit_h"](checkpoint=path)
    sam.eval()
    return sam.to(get_device())


def load_sam_adapter():
    # load the original SAM model
    net = load_sam_vit_h()
    net.eval()

    sam_weights = os.path.join(
        get_root_path(), 'data/external/sam_vit_h_4b8939.pth')    # load the original SAM weight
    with open(sam_weights, "rb") as f:
        state_dict = torch.load(f)
        new_state_dict = {k: v for k, v in state_dict.items(
        ) if k in net.state_dict() and net.state_dict()[k].shape == v.shape}
        net.load_state_dict(new_state_dict, strict=False)

    # load task-specific adapter
    adapter_path = os.path.join(
        get_root_path(), 'data/external/Melanoma_Photo_SAM_1024.pth')
    checkpoint_file = os.path.join(adapter_path)
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict
    net.load_state_dict(new_state_dict, strict=False)
    return net


def calculate_iou(pred_mask, true_mask):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    assert pred_mask.shape == true_mask.shape, \
        f"Shape mismatch: pred_mask.shape={pred_mask.shape}, true_mask.shape={true_mask.shape}"

    # 确保掩码是布尔类型
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0.0


def calculate_dice(pred_mask, true_mask):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    assert pred_mask.shape == true_mask.shape, \
        f"Shape mismatch: pred_mask.shape={pred_mask.shape}, true_mask.shape={true_mask.shape}"

    # 确保掩码是布尔类型
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    return (2 * intersection) / total if total != 0 else 0.0
