import os
import shutil
import re
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm

from helpers import extract_image_id, filter_images


# def extract_image_id(image_path):
#     '''从文件路径中提取 image_id'''
#     # 替换_mask_\d+.png 为 ''
#     return re.sub(r'_mask_\d+.png', '', os.path.basename(image_path)).split('.')[0]
# match = re.search(r'(.+)_mask_\d+.png', os.path.basename(image_path))
# if match:
#     return match.group(0)
# else:
#     raise ValueError(f"无法从路径中提取 image_id: {image_path}")


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 Dice 系数。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: Dice 系数
    """
    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)
    intersection = np.logical_and(mask, ground_truth)
    mask_sum = np.sum(mask)
    ground_truth_sum = np.sum(ground_truth)
    dice_score = (2 * np.sum(intersection)) / (mask_sum + ground_truth_sum)
    # 识别散点并降低奖励
    labeled_mask, num_features = label(mask, return_num=True)
    scatter_penalty = (num_features - 1) * 0.01  # 每个散点降低的奖励值，假设为0.01
    adjusted_dice_score = dice_score - scatter_penalty

    return max(adjusted_dice_score, 0)  # 确保奖励值不为负


def copy_best_rewards(in_dir, out_dir, ground_truth_dir, index, is_ground_truth=False):
    '''复制最佳奖励到文件夹,并命名为{image_id}_mask_{index}.png
    使用rewards_function重新计算奖励,并保存到{image_id}_mask_{index}_reward.txt'''
    input_dir = in_dir
    ground_truth_dir = ground_truth_dir
    output_dir = out_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    image_files.sort()
    if is_ground_truth:
        image_files = filter_images(image_files)
    for image_file in tqdm(image_files, desc="Copy best rewards"):
        image_id = extract_image_id(image_file)
        image_path = os.path.join(input_dir, image_file)
        # 匹配以image_id开头的ground_truth文件
        ground_truth_files = [f for f in os.listdir(
            ground_truth_dir) if f.startswith(image_id)]
        if len(ground_truth_files) == 0:
            print(f"Ground truth for {image_file} does not exist.")
            continue
        ground_truth_file = ground_truth_files[0]
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_file)

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
    output_dir = 'data/processed/train/expanded'
    ground_truth_dir = 'data/raw/train/ISBI2016_ISIC/ground_truth'
    copy_best_rewards(ground_truth_dir, output_dir, ground_truth_dir, 0)
    image_dir = 'data/processed/train/ISBI2016_ISIC/auto_masks/best_rewards'
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 1)
    image_dir = 'data/processed/train/ISBI2016_ISIC/connected_point_masks/best_rewards'
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 2)
    image_dir = 'data/processed/train/ISBI2016_ISIC/point_masks/best_rewards'
    copy_best_rewards(image_dir, output_dir, ground_truth_dir, 3)
