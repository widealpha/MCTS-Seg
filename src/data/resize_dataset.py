import os
import re
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm
from data.expand_dataset import extract_image_id
from utils.helpers import get_root_path


root_path = get_root_path()

def extract_mask_id(mask_file):
    '''从文件路径中提取符合 mask_\d+ 形式的 mask_id'''
    match = re.search(r'mask_\d+', os.path.basename(mask_file))
    if match:
        return match.group(0)
    else:
        raise ValueError(f"无法从路径中提取 mask_id: {mask_file}")


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 Dice 系数，并根据散点数量降低奖励。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: 调整后的 Dice 系数
    """
    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)
    intersection = np.logical_and(mask, ground_truth)
    mask_sum = np.sum(mask)
    ground_truth_sum = np.sum(ground_truth)
    dice_score = (2 * np.sum(intersection)) / \
        (mask_sum + ground_truth_sum)

    # 识别散点并降低奖励
    labeled_mask, num_features = label(mask, return_num=True)
    scatter_penalty = (num_features - 1) * 0.02  # 每个散点降低的奖励值，假设为0.02
    adjusted_dice_score = dice_score - scatter_penalty

    return max(adjusted_dice_score, 0)  # 确保奖励值不为负


def resize_and_compare_images(in_dir, out_dir, raw_dir, size=(1024, 1024)):
    """
    获取 in_dir 目录下的所有图片，并将它们处理到统一的大小，然后与 mask_0 进行对比计算奖励。
    同时从 raw_dir 中获取原始图像并进行相同的调整。
    :param in_dir: 输入图片目录
    :param out_dir: 输出图片目录
    :param raw_dir: 原始图片目录
    :param size: 处理后的图片大小
    """
    input_dir = os.path.join(root_path, in_dir)
    output_dir = os.path.join(root_path, out_dir)
    raw_image_dir = os.path.join(root_path, raw_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and '_mask_' in f]
    image_files.sort()
    for image_file in tqdm(image_files, desc="Resize and reward images"):
        image_id = extract_image_id(image_file)
        mask_id = extract_mask_id(image_file)
        image_path = os.path.join(input_dir, image_file)
        ground_truth_path = os.path.join(input_dir, f"{image_id}_mask_0.png")
        raw_image_path = os.path.join(raw_image_dir, f"{image_id}.jpg")

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {image_file} does not exist.")
            continue

        if not os.path.exists(raw_image_path):
            print(f"Raw image for {image_file} does not exist.")
            continue

        image = Image.open(image_path).convert('L')
        ground_truth = Image.open(ground_truth_path).convert('L')
        raw_image = Image.open(raw_image_path).convert('RGB')

        # Resize images
        image = image.resize(size)
        ground_truth = ground_truth.resize(size)
        raw_image = raw_image.resize(size, Image.LANCZOS)

        mask = np.array(image)
        ground_truth = np.array(ground_truth)

        reward = rewards_function(mask, ground_truth)

        output_image_path = os.path.join(output_dir, image_file)
        output_reward_path = os.path.join(output_dir, f"{image_id}_{mask_id}_reward.txt")
        output_raw_image_path = os.path.join(output_dir, f"{image_id}_raw.jpg")

        image.save(output_image_path)
        raw_image.save(output_raw_image_path)
        with open(output_reward_path, 'w') as f:
            f.write(f"{reward}\n")


if __name__ == '__main__':
    in_dir = 'data/processed/train/expanded'
    out_dir = 'data/processed/train/resized'
    raw_dir = 'data/raw/train/ISBI2016_ISIC/image'
    resize_and_compare_images(in_dir, out_dir, raw_dir, (1024, 1024))
