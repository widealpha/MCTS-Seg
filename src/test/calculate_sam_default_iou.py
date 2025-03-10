import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.helpers import get_data_path, get_mcts_path, get_root_path
from data.helpers import extract_image_id

data_path = get_data_path()
root_path = get_root_path()
mcts_path = get_mcts_path()


def calculate_iou(mask, ground_truth):
    mask = np.array(mask)
    ground_truth = np.array(ground_truth)
    intersection = np.logical_and(mask, ground_truth).sum()
    union = np.logical_or(mask, ground_truth).sum()
    if union == 0:
        return 0.0
    return intersection / union


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


def main():
    mode = 'test'
    ground_truth_dir = os.path.join(data_path, f'raw/{mode}/ground_truth')
    mcts = os.path.join(mcts_path)

    one_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_1/largest_connected')
    one_bg_one_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_2/largest_connected')
    one_bg_two_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_3/largest_connected')

    iou_mean = calculate_mean_iou(one_fg, ground_truth_dir)
    print(f"[Largest Connected] One Fg Mean IoU: {iou_mean}")
    iou_mean = calculate_mean_iou(one_bg_one_fg, ground_truth_dir)
    print(f"[Largest Connected] One Bg One Fg Mean IoU: {iou_mean}")
    iou_mean = calculate_mean_iou(one_bg_two_fg, ground_truth_dir)
    print(f"[Largest Connected] One Bg Two Fg Mean IoU: {iou_mean}")

    iou_mean = calculate_mean_iou(mcts, ground_truth_dir)
    print(f"MCTS Mean IoU: {iou_mean}")

    one_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_1/best_rewards')
    one_bg_one_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_2/best_rewards')
    one_bg_two_fg = os.path.join(
        data_path, f'processed/{mode}/random_point_masks_3/best_rewards')
    iou_mean = calculate_mean_iou(one_fg, ground_truth_dir)
    print(f"[Best Reward] One Fg Mean IoU: {iou_mean}")
    iou_mean = calculate_mean_iou(one_bg_one_fg, ground_truth_dir)
    print(f"[Best Reward] One Bg One Fg Mean IoU: {iou_mean}")
    iou_mean = calculate_mean_iou(one_bg_two_fg, ground_truth_dir)
    print(f"[Best Reward] One Bg Two Fg Mean IoU: {iou_mean}")


if __name__ == '__main__':
    # mcts = os.path.join(root_path, f'results/mcts-40s-3points')
    # ground_truth_dir = os.path.join(data_path, f'raw/test/ground_truth')
    # iou = calculate_mean_iou(mcts, ground_truth_dir)
    # print(f"MCTS Mean IoU: {iou}")
    main()
