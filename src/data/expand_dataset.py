import os
import shutil
import re
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm
from utils.helpers import get_root_path

root_path = get_root_path()


def extract_image_id(image_path):
    '''从文件路径中提取符合 ISIC_\d+ 形式的 image_id'''
    match = re.search(r'ISIC_\d+', os.path.basename(image_path))
    if match:
        return match.group(0)
    else:
        raise ValueError(f"无法从路径中提取 image_id: {image_path}")


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 Dice 系数。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: Dice 系数
    """
    intersection = np.logical_and(mask, ground_truth)
    dice_score = (2 * np.sum(intersection)) / \
        (np.sum(mask) + np.sum(ground_truth) + 1e-7)
    return dice_score


def copy_ground_truth(ground_truth_dir):
    '''复制ground truth到文件夹,并命名为{image_id}_mask_0.png'''
    output_dir = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/ground_truth')
    os.makedirs(output_dir, exist_ok=True)

    ground_truth_files = [f for f in os.listdir(
        ground_truth_dir) if f.endswith('.png')]
    for gt_file in ground_truth_files:
        image_id = extract_image_id(gt_file)
        src_path = os.path.join(ground_truth_dir, gt_file)
        dst_path = os.path.join(output_dir, f"{image_id}_mask_0.png")
        shutil.copy(src_path, dst_path)


def copy_best_rewards(image_dir, output_dir, ground_truth_dir, index):
    '''复制最佳奖励到文件夹,并命名为{image_id}_mask_{index}.png
    使用rewards_function重新计算奖励,并保存到{image_id}_mask_{index}_reward.txt'''
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort()
    for image_file in tqdm(image_files, desc="Processing images"):
        image_id = extract_image_id(image_file)
        image_path = os.path.join(image_dir, image_file)
        ground_truth_path = os.path.join(
            ground_truth_dir, f"{image_id}_Segmentation.png")

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {image_file} does not exist.")
            continue

        image = Image.open(image_path).convert('L')
        ground_truth = Image.open(ground_truth_path).convert('L')

        mask = np.array(image)
        ground_truth = np.array(ground_truth)

        reward = rewards_function(mask, ground_truth)

        output_image_path = os.path.join(
            output_dir, f"{image_id}_mask_{index}.png")
        output_reward_path = os.path.join(
            output_dir, f"{image_id}_mask_{index}_reward.txt")

        shutil.copy(image_path, output_image_path)
        with open(output_reward_path, 'w') as f:
            f.write(f"{reward}\n")


if __name__ == '__main__':
    output_dir = os.path.join(
        root_path, 'data/processed/train/expanded')
    ground_truth_dir = os.path.join(
        root_path, 'data/raw/train/ISBI2016_ISIC/ground_truth')
    copy_best_rewards(ground_truth_dir, output_dir, ground_truth_dir, 0)
    image_dir = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/auto_masks/best_rewards')
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 1)
    image_dir = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/connected_point_masks/best_rewards')
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 2)
    image_dir = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/point_masks/best_rewards')
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 3)
