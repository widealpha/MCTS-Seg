from functools import partial
import os
import platform
import random
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.tensorboard import SummaryWriter
from src.cfg import parse_args
from src.models.msa_predictor import MSAPredictor


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


def get_result_path():
    res = os.path.join(get_root_path(), 'results')
    os.makedirs(res, exist_ok=True)
    return res


def get_baseline_result_path():
    res = os.path.join(get_result_path(), 'baseline', get_dataset())
    os.makedirs(res, exist_ok=True)
    return res


def get_mcts_result_path(suffix=None):
    res = os.path.join(get_result_path(), 'mcts',
                       f'{get_dataset()}', )
    os.makedirs(res, exist_ok=True)
    return res


def get_ablation_result_path():
    res = os.path.join(get_result_path(), 'ablation', get_dataset())
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
    sam.eval().to(device=get_device())
    return sam


def load_sam_adapter():
    # load the original SAM model
    from src.baseline.msa.sam import sam_model_registry as msa_model_registry
    net = msa_model_registry['vit_h'](parse_args(), checkpoint=os.path.join(
        get_root_path(), 'data/external/sam_vit_h_4b8939.pth'))
    dataset = get_dataset()
    if dataset == 'ISIC2018':
        weight_name = 'msa_isic2018_512_vit_h.pth'
    elif dataset == 'ISIC2016':
        weight_name = 'msa_isic2016_512_vit_h.pth'
    # load task-specific adapter
    msa_ckpt = parse_args().msa_ckpt
    if msa_ckpt is None or msa_ckpt == '':
        weights_path = os.path.join(
            get_root_path(),
            f'data/external/{weight_name}')
    else:
        weights_path = parse_args().msa_ckpt

    print(f'=> resuming from {weights_path}')
    assert os.path.exists(weights_path)
    checkpoint_file = os.path.join(weights_path)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:0'
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict, strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    print(
        f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch}, best_tol {best_tol})')
    return net.eval().to(get_device())


def load_medsam():
    dataset = get_dataset()
    if dataset == 'ISIC2018':
        weight_name = 'medsam_isic2018_1014_vit_b.pth'
    elif dataset == 'ISIC2016':
        weight_name = 'medsam_isic2016_1024_vit_b.pth'
    # load task-specific adapter
    weights_path = os.path.join(
        get_root_path(),
        f'data/external/{weight_name}')
    path = os.path.join(
        get_root_path(), 'src/baseline/MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
    torch.save(torch.load(path, map_location='cpu')['model'], weights_path)

    sam = sam_model_registry["vit_b"](checkpoint=weights_path)
    sam.eval().to(device=get_device())
    return sam


def calculate_iou(pred_mask, true_mask, threshold=0.5):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    assert pred_mask.shape == true_mask.shape, \
        f"Shape mismatch: pred_mask.shape={pred_mask.shape}, true_mask.shape={true_mask.shape}"

    # 确保掩码是布尔类型
    pred_mask = pred_mask > threshold
    true_mask = true_mask > threshold

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0.0


def calculate_dice(pred_mask, true_mask, threshold=0.5):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    assert pred_mask.shape == true_mask.shape, \
        f"Shape mismatch: pred_mask.shape={pred_mask.shape}, true_mask.shape={true_mask.shape}"

    # 确保掩码是布尔类型
    pred_mask = pred_mask > threshold
    true_mask = true_mask > threshold

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    return (2 * intersection) / total if total != 0 else 0.0


def set_chinese_font():
    matplotlib.use('TkAgg')

    # 常见中文字体按平台列出
    font_candidates = {
        'Windows': ['Microsoft YaHei', 'SimHei'],
        'Darwin': ['PingFang SC', 'Heiti SC'],
        'Linux': ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'AR PL UKai CN']
    }

    system = platform.system()
    fonts_available = set(f.name for f in fm.fontManager.ttflist)
    print(f"当前系统：{system}")
    print(f"系统中检测到的字体数量：{len(fonts_available)}")

    for font in font_candidates.get(system, []):
        if font in fonts_available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已设置 matplotlib 中文字体为：{font}")
            return

    print("⚠️ 未找到常用中文字体，请考虑手动安装（如 Noto Sans CJK 或 SimHei）")
