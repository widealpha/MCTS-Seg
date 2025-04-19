import os
import shutil

import numpy as np
from tqdm import tqdm
from PIL import Image

from src.cfg import parse_args
from src.preprocess.helpers import extract_image_id, extract_mask_id, filter_images
from src.utils.helpers import calculate_dice, calculate_iou, get_data_path, get_root_path, setup_seed
from sam_seg import sam_baseline_auto_mask, sam_baseline_point_mask, sam_point_mask, sam_random_point_mask


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 IOU，并根据散点数量降低奖励。
    然后将品质按照不同区间划分成 12 档，将奖励映射到 0-11 之间。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: 调整后的奖励值
    """
    reward = calculate_dice(mask, ground_truth)
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

    # 对所有 reward 进行zscore并保存
    normalized_rewards = [(r - mean_reward) /
                          std_reward for r in reward_values]
    # min_reward = min(normalized_rewards)
    # max_reward = max(normalized_rewards)
    # normalized_rewards = [(r - min_reward) /
    #                       (max_reward - min_reward) for r in normalized_rewards]
    for (reward, image_id, mask_id), norm_reward in zip(rewards, normalized_rewards):
        norm_reward_path = os.path.join(
            output_dir, f"{image_id}_mask_{mask_id}_normalized_reward.txt")
        with open(norm_reward_path, 'w') as f:
            f.write(f"{norm_reward}\n")

    # 保存全局的均值和标准差
    stats_path = os.path.join(output_dir, "reward_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Mean: {mean_reward}\n")
        f.write(f"Std: {std_reward}\n")

    return mean_reward, std_reward


def resize_and_compare_images(in_dir, out_dir, raw_dir, size, train_mean=None, train_std=None):
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

    mask_files = [f for f in os.listdir(
        input_dir) if f.endswith('.png') and '_mask_' in f]
    mask_files.sort()
    raw_image_map = {}
    for image in os.listdir(raw_image_dir):
        image_id = extract_image_id(image)
        raw_image_map[image_id] = image
    for mask_file in tqdm(mask_files, desc="Resize and reward images"):
        image_id = extract_image_id(mask_file)
        mask_id = extract_mask_id(mask_file)
        mask_path = os.path.join(input_dir, mask_file)
        ground_truth_path = os.path.join(input_dir, f"{image_id}_mask_0.png")
        if image_id not in raw_image_map:
            print(f"Raw image for {mask_file} does not exist.")
            continue
        raw_image_file = raw_image_map[image_id]
        raw_image_path = os.path.join(raw_image_dir, raw_image_file)

        mask = Image.open(mask_path).convert('L')
        ground_truth = Image.open(ground_truth_path).convert('L')
        raw_image = Image.open(raw_image_path).convert('RGB')

        # Resize images
        if size is not None:
            if mask.size != size:
                mask = mask.resize(size, Image.NEAREST)
            if ground_truth.size != size:
                ground_truth = ground_truth.resize(size, Image.NEAREST)
            if raw_image.size != size:
                raw_image = raw_image.resize(size, Image.BICUBIC)

        mask_array = np.array(mask)
        ground_truth = np.array(ground_truth)

        reward = rewards_function(mask_array, ground_truth)
        rewards.append((reward, image_id, mask_id))

        output_image_path = os.path.join(output_dir, mask_file)
        output_reward_path = os.path.join(
            output_dir, f"{image_id}_mask_{mask_id}_reward.txt")
        output_raw_image_path = os.path.join(output_dir, f"{image_id}_raw.png")

        mask.save(output_image_path)
        raw_image.save(output_raw_image_path)
        with open(output_reward_path, 'w') as f:
            f.write(f"{reward}\n")

        # iou = calculate_iou(mask, ground_truth)
        # # Save IoU result
        # iou_result_path = os.path.join(
        #     output_dir, f"{image_id}_mask_{mask_id}_iou.txt")
        # with open(iou_result_path, 'w') as f:
        #     f.write(f"{iou}\n")
    if train_mean is None or train_std is None:
        train_mean, train_std = normalize_rewards(rewards, out_dir)
        print(f"Mean: {train_mean}, Std: {train_std}")
        return train_mean, train_std
    else:
        normalize_param_path = os.path.join(out_dir, "normalize_param.txt")
        with open(normalize_param_path, 'w') as f:
            f.write(f"Mean: {train_mean}\n")
            f.write(f"Std: {train_std}\n")
        train_mean, train_std = normalize_rewards(
            rewards, out_dir, train_mean, train_std)
        print(f"Mean: {train_mean}, Std: {train_std}")
        return train_mean, train_std


def copy_dir(in_dir, out_dir, index):
    input_dir = in_dir
    output_dir = out_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    image_files.sort()
    image_files = filter_images(image_files)
    for image_file in tqdm(image_files, desc=f"Copy {index} masks"):
        image_id = extract_image_id(image_file)
        image_path = os.path.join(input_dir, image_file)
        output_image_path = os.path.join(
            output_dir, f"{image_id}_mask_{index}.png")

        shutil.copy(image_path, output_image_path)


def generate_data():
    data_types = ['train', 'test']
    train_mean = None
    train_std = None
    for data_type in data_types:
        data_path = get_data_path()

        raw_path = os.path.join(data_path, 'raw')
        processed_path = os.path.join(data_path, 'processed')
        final_path = os.path.join(data_path, 'final')

        raw_image_dir = os.path.join(raw_path, data_type, 'image')
        ground_truth_dir = os.path.join(raw_path, data_type, 'gt')

        random_point_masks_dir = [(i, os.path.join(
            processed_path, data_type, f'random_point_masks_{i}')) for i in range(1, 4)]
        # 对point_masks_dir中最佳数据取最大联通分量mask的目录
        center_point_masks_dir = os.path.join(
            processed_path, data_type, 'center')
        # 整合ground_truth以及上述三/四种mask的目录
        augmented_dir = os.path.join(processed_path, data_type, 'augmented')
        # 对上述数据应用新的reward算法并缩放的保存结果的目录
        final_dir = os.path.join(final_path, data_type)

        print(f"Generating data for {data_type} set")

        # for i_dir in random_point_masks_dir:
        #     sam_random_point_mask(fg_point_num=(i_dir[0] - 1) // 2 + 1, bg_point_num=i_dir[0] // 2,
        #                           in_dir=raw_image_dir, out_dir=i_dir[1], ground_truth_dir=ground_truth_dir)

        # copy_dir(in_dir=ground_truth_dir, out_dir=augmented_dir, index=0)
        # for i_dir in random_point_masks_dir:
        #     copy_dir(in_dir=os.path.join(
        #         i_dir[1]), out_dir=augmented_dir, index=i_dir[0])

        args = parse_args()
        image_size = (args.image_size, args.image_size)

        # image_size = None
        if train_mean == None or train_std == None:
            # 调整图像大小并生成奖励
            train_mean, train_std = resize_and_compare_images(
                in_dir=augmented_dir, out_dir=final_dir, raw_dir=raw_image_dir, size=image_size)
        else:
            resize_and_compare_images(
                in_dir=augmented_dir, out_dir=final_dir, raw_dir=raw_image_dir, size=image_size, train_mean=train_mean, train_std=train_std)


if __name__ == '__main__':
    setup_seed()
    generate_data()
