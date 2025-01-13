import os
import shutil
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops
from tqdm import tqdm
from utils.helpers import get_root_path


root_path = get_root_path()


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
    dice_score = (2 * np.sum(intersection)) / \
        (np.sum(mask) + np.sum(ground_truth) + 1e-7)
    return dice_score
    # """
    # 计算 mask 和 ground_truth 之间的 IoU 和 F1 score 的调和平均值。
    # :param mask: 预测的 mask
    # :param ground_truth: ground truth mask
    # :return: IoU 和 F1 score 的调和平均值
    # """
    # intersection = np.logical_and(mask, ground_truth)
    # union = np.logical_or(mask, ground_truth)
    # iou_score = np.sum(intersection) / np.sum(union)

    # # 计算 F1 score
    # true_positive = np.sum(intersection)
    # false_positive = np.sum(mask) - true_positive
    # false_negative = np.sum(ground_truth) - true_positive

    # precision = true_positive / (true_positive + false_positive + 1e-7)
    # recall = true_positive / (true_positive + false_negative + 1e-7)
    # f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    # # 计算 IoU 和 F1 score 的调和平均值
    # harmonic_mean = 2 * (iou_score * f1_score) / (iou_score + f1_score + 1e-7)

    # return harmonic_mean


def select_best_rewards_image(in_dir='data/processed/train/ISBI2016_ISIC/auto_masks',
                              out_dir='data/processed/train/ISBI2016_ISIC/auto_masks/best_rewards',
                              ground_truth_dir='data/raw/train/ISBI2016_ISIC/ground_truth'):
    """
    从 data/raw/train/ISBI2016_ISIC/ground_truth 中读取 ground_truth 文件，
    遍历 data/processed/train/ISBI2016_ISIC 中的所有 image_id 相同的 mask，
    获取到每个 image_id 中 IoU 最大的 mask，并将图片及其对应的 IoU 保存到
    data/processed/train/best_iou 文件夹下。
    """
    ground_truth_dir = os.path.join(
        root_path, ground_truth_dir)
    processed_dir = os.path.join(
        root_path, in_dir)
    best_rewards_dir = os.path.join(root_path, out_dir)

    if not os.path.exists(best_rewards_dir):
        os.makedirs(best_rewards_dir)

    ground_truth_files = os.listdir(ground_truth_dir)
    ground_truth_files.sort()

    for gt_file in tqdm(ground_truth_files, desc="Processing ground truth images"):
        image_id = gt_file.replace('_Segmentation.png', '')
        ground_truth_path = os.path.join(ground_truth_dir, gt_file)
        ground_truth = Image.open(ground_truth_path).convert('L')
        ground_truth = np.array(ground_truth)

        best_score = -np.inf
        best_mask_file = None

        for file_name in os.listdir(processed_dir):
            if file_name.startswith(image_id) and file_name.endswith('.png'):
                mask_path = os.path.join(processed_dir, file_name)
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask)

                score = rewards_function(mask, ground_truth)

                if score > best_score:
                    best_score = score
                    best_mask_file = file_name

        if best_mask_file:
            best_mask_path = os.path.join(processed_dir, best_mask_file)
            best_mask = Image.open(best_mask_path)
            best_mask.save(os.path.join(best_rewards_dir, best_mask_file))
            with open(os.path.join(best_rewards_dir, f"{image_id}_best_score.txt"), 'w') as f:
                f.write(f"{best_score}\n")


def select_best_rewards_all_image(in_dir, ground_truth_dir, out_dir):
    """
    遍历每个子文件夹中的所有图片，并与对应的 ground truth 进行对比，选择奖励值最高的图片。
    :param image_dir: 存储图片的主文件夹
    :param ground_truth_dir: 存储 ground truth 的文件夹
    :param output_dir: 存储最佳奖励图片的文件夹
    """

    sub_dirs = [d for d in os.listdir(
        in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    sub_dirs.sort()
    os.makedirs(out_dir, exist_ok=True)
    for sub_dir in tqdm(sub_dirs, desc="Processing subdirectories"):
        image_id = sub_dir
        sub_dir_path = os.path.join(in_dir, sub_dir)
        ground_truth_path = os.path.join(
            ground_truth_dir, f"{image_id}_Segmentation.png")

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {image_id} does not exist.")
            continue

        ground_truth = Image.open(ground_truth_path).convert('L')
        ground_truth = np.array(ground_truth)

        best_reward = -np.inf
        best_image_file = None

        image_files = [f for f in os.listdir(
            sub_dir_path) if f.endswith('.png')]
        for image_file in image_files:
            image_path = os.path.join(sub_dir_path, image_file)
            image = Image.open(image_path).convert('L')
            mask = np.array(image)

            reward = rewards_function(mask, ground_truth)
            if reward > best_reward:
                best_reward = reward
                best_image_file = image_file

        if best_image_file:
            best_image_path = os.path.join(sub_dir_path, best_image_file)
            output_image_path = os.path.join(out_dir, f"{image_id}_best.png")
            shutil.copy(best_image_path, output_image_path)
            with open(os.path.join(out_dir, f"{image_id}_reward.txt"), 'w') as f:
                f.write(f"{best_reward}\n")


if __name__ == '__main__':
    dir_name = 'connected_point_masks'
    # select_best_rewards_image(in_dir=f'data/processed/train/ISBI2016_ISIC/{dir_name}',
    #                           out_dir=f'data/processed/train/ISBI2016_ISIC/{dir_name}/best_rewards')
    select_best_rewards_all_image(in_dir=f'data/processed/train/ISBI2016_ISIC/all_point_masks',
                                  out_dir=f'data/processed/train/ISBI2016_ISIC/all_point_masks/best_rewards',
                                  ground_truth_dir='data/raw/train/ISBI2016_ISIC/ground_truth')
