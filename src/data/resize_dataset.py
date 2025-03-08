import os
import re
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm
from data.expand_dataset import extract_image_id
from scipy.spatial.distance import directed_hausdorff, cdist


def extract_mask_id(mask_file):
    '''从文件路径中提取符合 mask_\d+ 形式的 mask_id'''
    match = re.search(r'mask_\d+', os.path.basename(mask_file))
    if match:
        return match.group(0)
    else:
        raise ValueError(f"无法从路径中提取 mask_id: {mask_file}")


def calculate_iou(mask, ground_truth):
    """
    计算两个掩码之间的交并比（IOU）。
    :param mask: 第一个掩码
    :param ground_truth: 第二个掩码
    :return: IOU 值
    """
    intersection = np.logical_and(mask, ground_truth).sum()
    union = np.logical_or(mask, ground_truth).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return (2 * intersection) / (mask1.sum() + mask2.sum() + 1e-7)


def hausdorff_distance(mask1, mask2):
    points1 = np.column_stack(np.where(mask1 > 0))
    points2 = np.column_stack(np.where(mask2 > 0))

    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]

    return max(d1, d2)


def average_surface_distance(mask1, mask2):
    points1 = np.column_stack(np.where(mask1 > 0))
    points2 = np.column_stack(np.where(mask2 > 0))

    d1 = cdist(points1, points2).min(axis=1)
    d2 = cdist(points2, points1).min(axis=1)

    return (d1.mean() + d2.mean()) / 2


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 IOU，并根据散点数量降低奖励。
    然后将品质按照不同区间划分成 12 档，将奖励映射到 0-11 之间。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: 调整后的奖励值
    """
    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)
    reward = average_surface_distance(mask, ground_truth)
    # 计算 IOU
    # iou = calculate_iou(mask, ground_truth)

    # 识别散点并降低奖励
    # labeled_mask, num_features = label(mask, return_num=True)
    # scatter_penalty = (num_features - 1) * 0.02  # 每个散点降低的奖励值，假设为0.02
    # adjusted_iou = iou - scatter_penalty

    # reward = max(adjusted_iou, 0)

    return reward


def normalize_rewards(rewards, output_dir, mean_reward=None, std_reward=None):
    # 提取奖励值
    reward_values = [r[0] for r in rewards]

    # 计算全局的均值和标准差
    if mean_reward is None or std_reward is None:
        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values)

    # 对所有 reward 进行归一化并保存
    normalized_rewards = [(r - mean_reward) /
                          std_reward for r in reward_values]
    for (reward, image_id, mask_id), norm_reward in zip(rewards, normalized_rewards):
        norm_reward_path = os.path.join(
            output_dir, f"{image_id}_{mask_id}_normalized_reward.txt")
        with open(norm_reward_path, 'w') as f:
            f.write(f"{norm_reward}\n")

    # 保存全局的均值和标准差
    stats_path = os.path.join(output_dir, "reward_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Mean: {mean_reward}\n")
        f.write(f"Std: {std_reward}\n")

    return mean_reward, std_reward


def normalize_test_rewards(rewards, train_mean, train_std, output_dir, normalize='minmax'):
    # 提取奖励值
    reward_values = [r[0] for r in rewards]
    normalized_rewards = [r for r in reward_values]

    if normalize == 'minmax':
        # 计算最小值和最大值
        min_reward = min(reward_values)
        max_reward = max(reward_values)

        # 避免除零错误
        if max_reward == min_reward:
            normalized_rewards = [0.5] * \
                len(reward_values)  # 如果所有奖励相同，全部设为 0.5
        else:
            normalized_rewards = [
                (r - min_reward) / (max_reward - min_reward) for r in reward_values]
    elif normalize == 'zscore':
        # 使用训练集的均值和标准差对测试集的奖励进行归一化
        normalized_rewards = [
            (r - train_mean) / train_std for r in reward_values]

    for (reward, image_id, mask_id), norm_reward in zip(rewards, normalized_rewards):
        norm_reward_path = os.path.join(
            output_dir, f"{image_id}_{mask_id}_normalized_reward.txt")
        with open(norm_reward_path, 'w') as f:
            f.write(f"{norm_reward}\n")


def resize_and_compare_images(in_dir, out_dir, raw_dir, size=(1024, 1024), train_mean=None, train_std=None):
    """
    获取 in_dir 目录下的所有图片，并将它们处理到统一的大小，然后与 mask_0 进行对比计算奖励。
    同时从 raw_dir 中获取原始图像并进行相同的调整。
    :param in_dir: 输入图片目录
    :param out_dir: 输出图片目录
    :param raw_dir: 原始图片目录
    :param size: 处理后的图片大小
    :param train_mean: 训练集的均值（仅在测试模式下使用）
    :param train_std: 训练集的标准差（仅在测试模式下使用）
    """
    input_dir = in_dir
    output_dir = out_dir
    raw_image_dir = raw_dir
    os.makedirs(output_dir, exist_ok=True)

    rewards = []

    image_files = [f for f in os.listdir(
        input_dir) if f.endswith('.png') and '_mask_' in f]
    image_files.sort()
    for image_file in tqdm(image_files, desc="Resize and reward images"):
        image_id = extract_image_id(image_file)
        mask_id = extract_mask_id(image_file)
        image_path = os.path.join(input_dir, image_file)
        ground_truth_path = os.path.join(input_dir, f"{image_id}_mask_0.png")
        # 匹配以image_id开头的raw image
        raw_image_files = [f for f in os.listdir(
            raw_image_dir) if f.startswith(image_id)]
        if len(raw_image_files) == 0:
            print(f"Raw image for {image_file} does not exist.")
            continue
        raw_image_file = raw_image_files[0]
        raw_image_path = os.path.join(raw_image_dir, raw_image_file)

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
        if size is not None:
            image = image.resize(size)
            ground_truth = ground_truth.resize(size)
            raw_image = raw_image.resize(size, Image.LANCZOS)

        mask = np.array(image)
        ground_truth = np.array(ground_truth)

        reward = rewards_function(mask, ground_truth)
        rewards.append((reward, image_id, mask_id))

        output_image_path = os.path.join(output_dir, image_file)
        output_reward_path = os.path.join(
            output_dir, f"{image_id}_{mask_id}_reward.txt")
        output_raw_image_path = os.path.join(output_dir, f"{image_id}_raw.jpg")

        image.save(output_image_path)
        raw_image.save(output_raw_image_path)
        with open(output_reward_path, 'w') as f:
            f.write(f"{reward}\n")

        iou = calculate_iou(mask, ground_truth)
        # Save IoU result
        iou_result_path = os.path.join(
            output_dir, f"{image_id}_{mask_id}_iou.txt")
        with open(iou_result_path, 'w') as f:
            f.write(f"{iou}\n")
    if train_mean is None or train_std is None:
        train_mean, train_std = normalize_rewards(rewards, out_dir)
        return train_mean, train_std
    else:
        normalize_rewards(rewards, out_dir, train_mean, train_std)


if __name__ == '__main__':
    train_mode = 'train'
    test_mode = 'test'
    train_in_dir = f'data/processed/{train_mode}/expanded'
    train_out_dir = f'data/processed/{train_mode}/resized'
    train_raw_dir = f'data/raw/{train_mode}/ISBI2016_ISIC/image'
    test_in_dir = f'data/processed/{test_mode}/expanded'
    test_out_dir = f'data/processed/{test_mode}/resized'
    test_raw_dir = f'data/raw/{test_mode}/ISBI2016_ISIC/image'

    train_mean, train_std = resize_and_compare_images(
        train_in_dir, train_out_dir, train_raw_dir, (1024, 1024), train_mode)
    resize_and_compare_images(test_in_dir, test_out_dir, test_raw_dir,
                              (1024, 1024), test_mode, train_mean, train_std)
