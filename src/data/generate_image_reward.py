import os

import numpy as np
from src.utils.helpers import get_root_path
from skimage.measure import label, regionprops
from PIL import Image


root_path = get_root_path()


def copy_image():
    pass


def select_best_rewards(image_id):
    """
    根据提供的 image_id 遍历图像列表，读取图像并计算奖励。
    :param image_id: 图像的 ID，例如 'ISIC_00000000'
    :return: 最佳奖励值
    """
    image_files = get_process_images(image_id)
    best_reward = -np.inf
    best_image = None

    for image_file in image_files:
        image_path = os.path.join(
            root_path, 'data/processed/train/ISBI2016_ISIC', image_file)
        ground_truth_path = image_path.replace('.png', '_mask.png')

        image = Image.open(image_path)
        ground_truth = Image.open(ground_truth_path)

        reward = calculate_reward(image, ground_truth)
        if reward > best_reward:
            best_reward = reward
            best_image = image_file

    return best_image, best_reward


def calculate_reward(image, ground_truth):
    """
    计算图像的奖励值。
    :param image: 原始图像
    :param ground_truth: 对应的 ground truth 图像
    :return: 奖励值
    """
    # 计算图像的整体性
    labeled_image = label(image)
    regions = regionprops(labeled_image)
    scatter_score = len(regions)

    # 计算图像和 ground truth 的 IoU
    intersection = np.logical_and(image, ground_truth)
    union = np.logical_or(image, ground_truth)
    iou_score = np.sum(intersection) / np.sum(union)

    # 计算最终奖励，整体性越高（散点越少）奖励越高，IoU 越高奖励越高
    reward = iou_score - scatter_score * 0.01  # 假设散点的权重为 0.01

    return reward


def get_process_images(image_id):
    """
    根据提供的 image_id 返回相应的一组图像。
    :param image_id: 图像的 ID，例如 'ISIC_00000000'
    :param directory: 存储图像的目录路径
    :return: 图像文件名列表
    """
    directory = os.path.join(root_path, 'data/processed/train/ISBI2016_ISIC')
    image_files = []
    for file_name in os.listdir(directory):
        if file_name.startswith(image_id):
            image_files.append(file_name)
    return image_files

if __name__ == 'main':
    pass