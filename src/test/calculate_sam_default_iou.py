import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.helpers import get_data_path, get_log_path, get_mcts_path, get_root_path
from data.helpers import extract_image_id

data_path = get_data_path()
root_path = get_root_path()
mcts_path = get_mcts_path()
log_path = get_log_path()


def calculate_iou(mask, ground_truth):
    mask = np.array(mask).astype(bool)
    ground_truth = np.array(ground_truth).astype(bool)
    intersection = np.logical_and(mask, ground_truth).sum()
    union = np.logical_or(mask, ground_truth).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_dice(mask, ground_truth):
    mask = np.array(mask).astype(bool)
    ground_truth = np.array(ground_truth).astype(bool)
    intersection = np.logical_and(mask, ground_truth).sum()
    mask_sum = mask.sum()
    ground_truth_sum = ground_truth.sum()
    if mask_sum + ground_truth_sum == 0:
        return 1.0
    return 2 * intersection / (mask_sum + ground_truth_sum)


def calculate_variance(values):
    return np.var(values)


def calculate_mean_iou(mask_dir, ground_truth_dir):
    iou_results = []
    images = os.listdir(ground_truth_dir)
    images.sort()
    image_ids = [extract_image_id(image) for image in images]

    for image, image_id in tqdm(zip(images, image_ids), total=len(images)):
        ground_truth_path = os.path.join(ground_truth_dir, image)
        ground_truth = Image.open(ground_truth_path).convert('L')

        # 使用正则表达式匹配符合条件的文件
        mask_files = [f for f in os.listdir(mask_dir) if re.match(
            f"{image_id}_mask(_\d)*.png", f)]
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # 调整 ground_truth 的大小与 mask 一致
                ground_truth_resized = ground_truth.resize(
                    mask.size, Image.NEAREST)
                mask = np.array(mask)
                ground_truth_resized = np.array(ground_truth_resized)
                iou = calculate_iou(mask, ground_truth_resized)
                iou_results.append(iou)

    mean_iou = np.mean(iou_results)
    return mean_iou


def calculate_mean_and_variance_iou(mask_dir, ground_truth_dir, log_file="iou_log.txt"):
    iou_results = []
    images = os.listdir(ground_truth_dir)
    images.sort()
    image_ids = [extract_image_id(image) for image in images]

    with open(os.path.join(mask_dir, log_file), "w") as log:
        for image, image_id in tqdm(zip(images, image_ids), total=len(images)):
            ground_truth_path = os.path.join(ground_truth_dir, image)
            ground_truth = Image.open(ground_truth_path).convert('L')

            # 使用正则表达式匹配符合条件的文件
            mask_files = [f for f in os.listdir(mask_dir) if re.match(
                f"{image_id}_mask(_\d)*.png", f)]
            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    # 调整 ground_truth 的大小与 mask 一致
                    ground_truth_resized = ground_truth.resize(
                        mask.size, Image.NEAREST)
                    mask = np.array(mask)
                    ground_truth_resized = np.array(ground_truth_resized)
                    iou = calculate_iou(mask, ground_truth_resized)
                    iou_results.append(iou)

                    # 如果 IoU 小于 0.5，写入日志
                    if iou < 0.5:
                        log.write(
                            f"Low IoU: {iou:.4f}, Image ID: {image_id}, Mask File: {mask_file}\n")

    mean_iou = np.mean(iou_results)
    variance_iou = calculate_variance(iou_results)
    std_dev_iou = np.std(iou_results)  # 添加标准差计算
    return mean_iou, variance_iou, std_dev_iou


def calculate_mean_dice(mask_dir, ground_truth_dir):
    dice_results = []
    images = os.listdir(ground_truth_dir)
    images.sort()
    image_ids = [extract_image_id(image) for image in images]

    for image, image_id in tqdm(zip(images, image_ids), total=len(images)):
        ground_truth_path = os.path.join(ground_truth_dir, image)
        ground_truth = Image.open(ground_truth_path).convert('L')

        # 使用正则表达式匹配符合条件的文件
        mask_files = [f for f in os.listdir(mask_dir) if re.match(
            f"{image_id}_mask(_\d)*.png", f)]
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # 调整 ground_truth 的大小与 mask 一致
                ground_truth_resized = ground_truth.resize(
                    mask.size, Image.NEAREST)
                mask = np.array(mask)
                ground_truth_resized = np.array(ground_truth_resized)
                dice = calculate_dice(mask, ground_truth_resized)
                dice_results.append(dice)

    mean_dice = np.mean(dice_results)
    return mean_dice


def calculate_mean_and_variance_dice(mask_dir, ground_truth_dir, log_file="dice_log.txt"):
    dice_results = []
    images = os.listdir(ground_truth_dir)
    images.sort()
    image_ids = [extract_image_id(image) for image in images]

    with open(os.path.join(mask_dir, log_file), "w") as log:
        for image, image_id in tqdm(zip(images, image_ids), total=len(images)):
            ground_truth_path = os.path.join(ground_truth_dir, image)
            ground_truth = Image.open(ground_truth_path).convert('L')

            # 使用正则表达式匹配符合条件的文件
            mask_files = [f for f in os.listdir(mask_dir) if re.match(
                f"{image_id}_mask(_\d)*.png", f)]
            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    # 调整 ground_truth 的大小与 mask 一致
                    ground_truth_resized = ground_truth.resize(
                        mask.size, Image.NEAREST)
                    mask = np.array(mask)
                    ground_truth_resized = np.array(ground_truth_resized)
                    dice = calculate_dice(mask, ground_truth_resized)
                    dice_results.append(dice)

                    # 如果 Dice 小于 0.6，写入日志
                    if dice < 0.6:
                        log.write(
                            f"Low Dice: {dice:.4f}, Image ID: {image_id}, Mask File: {mask_file}\n")

    mean_dice = np.mean(dice_results)
    variance_dice = calculate_variance(dice_results)
    std_dev_dice = np.std(dice_results)  # 添加标准差计算
    return mean_dice, variance_dice, std_dev_dice


def calculate_score(mask_dir, ground_truth_dir, score_type='iou'):
    if score_type == 'iou':
        calculate_mean = calculate_mean_iou(mask_dir, ground_truth_dir)
    elif score_type == 'dice':
        calculate_mean = calculate_mean_dice(mask_dir, ground_truth_dir)
    return calculate_mean


def calculate_score_with_variance(mask_dir, ground_truth_dir, score_type='iou'):
    if score_type == 'iou':
        mean, variance, std_dev = calculate_mean_and_variance_iou(
            mask_dir, ground_truth_dir)
    elif score_type == 'dice':
        mean, variance, std_dev = calculate_mean_and_variance_dice(
            mask_dir, ground_truth_dir)
    return mean, variance, std_dev


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def main(split, score_type, dataset, data_path, mcts_path):
    # split = 'test'
    # score_type = 'dice'
    print(f"Current split {split}")
    print(f"Current dataset {dataset}")
    print(f"Current score type {score_type}")
    ground_truth_dir = os.path.join(data_path, f'raw/{split}/ground_truth')
    mcts = os.path.join(rreplace(mcts_path, "mcts", 'mcts_reseg', 1))

    # one_fg = os.path.join(
    #     data_path, f'processed/{split}/random_point_masks_1/largest_connected')
    # one_bg_one_fg = os.path.join(
    #     data_path, f'processed/{split}/random_point_masks_2/largest_connected')
    # one_bg_two_fg = os.path.join(
    #     data_path, f'processed/{split}/random_point_masks_3/largest_connected')

    # score_mean, score_variance = calculate_score_with_variance(one_fg, ground_truth_dir, score_type)
    # print(f"[Largest Connected] One Fg Mean {score_type}: {score_mean}, Variance: {score_variance}")
    # score_mean, score_variance = calculate_score_with_variance(one_bg_one_fg, ground_truth_dir, score_type)
    # print(f"[Largest Connected] One Bg One Fg Mean {score_type}: {score_mean}, Variance: {score_variance}")
    # score_mean, score_variance = calculate_score_with_variance(one_bg_two_fg, ground_truth_dir, score_type)
    # print(f"[Largest Connected] One Bg Two Fg Mean {score_type}: {score_mean}, Variance: {score_variance}")

    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        mcts, ground_truth_dir, score_type)
    print(
        f"MCTS Mean {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")

    one_fg = os.path.join(
        data_path, f'processed/{split}/random_point_masks_1/best_rewards')
    one_bg_one_fg = os.path.join(
        data_path, f'processed/{split}/random_point_masks_2/best_rewards')
    one_bg_two_fg = os.path.join(
        data_path, f'processed/{split}/random_point_masks_3/best_rewards')
    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        one_fg, ground_truth_dir, score_type)
    print(
        f"[Best Reward] One Fg Mean {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")
    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        one_bg_one_fg, ground_truth_dir, score_type)
    print(
        f"[Best Reward] One Bg One Fg Mean {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")
    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        one_bg_two_fg, ground_truth_dir, score_type)
    print(
        f"[Best Reward] One Bg Two Fg Mean {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")

    point_baseline = os.path.join(
        data_path, f'processed/{split}/baseline/point')
    auto_baseline = os.path.join(
        data_path, f'processed/{split}/baseline/auto')
    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        point_baseline, ground_truth_dir, score_type)
    print(
        f"[Baseline] Center Point baseline {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")
    score_mean, score_variance, score_std_dev = calculate_score_with_variance(
        auto_baseline, ground_truth_dir, score_type)
    print(
        f"[Baseline] Auto Point baseline {score_type}: {score_mean}, Variance: {score_variance}, Std Dev: {score_std_dev}")


if __name__ == '__main__':
    # mcts = os.path.join(root_path, f'results/mcts-40s-3points')
    # ground_truth_dir = os.path.join(data_path, f'raw/test/ground_truth')
    # iou = calculate_mean_iou(mcts, ground_truth_dir)
    # print(f"MCTS Mean IoU: {iou}")
    mcts_base_dir = os.path.join(root_path, 'result', 'mcts')
    data_base_dir = os.path.join(get_root_path(), 'data')
    data_path = os.path.join(data_base_dir, 'ISIC2016')
    mcts_path = os.path.join(mcts_base_dir, 'ISIC2016')
    main('test', 'iou', 'test', data_path, mcts_path)
    main('test', 'dice', 'test', data_path, mcts_path)
