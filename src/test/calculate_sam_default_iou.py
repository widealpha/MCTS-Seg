import os
import json
import numpy as np
from PIL import Image

from utils.helpers import get_root_path
root_path = get_root_path()


def calculate_iou(mask, ground_truth):
    mask = np.array(mask)
    ground_truth = np.array(ground_truth)
    intersection = np.logical_and(mask, ground_truth).sum()
    union = np.logical_or(mask, ground_truth).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    metadata_path = os.path.join(
        root_path, 'data/processed/test/ISBI2016_ISIC/auto_masks', 'mask_metadata.json')
    ground_truth_dir = os.path.join(
        root_path, 'data/raw/test/ISBI2016_ISIC/ground_truth')
    auto_masks_dir = os.path.join(
        root_path, 'data/processed/test/ISBI2016_ISIC/auto_masks')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    iou_results = []
    images = {}
    for item in metadata:
        image_id = item['image_file'].replace('.jpg', '')
        if image_id not in images:
            images[image_id] = []
        images[image_id].append(item)

    for image_id, items in images.items():
        ground_truth_path = os.path.join(
            ground_truth_dir, image_id + '_Segmentation.png')
        ground_truth = Image.open(ground_truth_path).convert('1')
        ground_truth = np.array(ground_truth)
        best_iou = 0.0
        first_image = None
        for item in items:
            if item['predicted_iou'] > best_iou:
                best_iou = item['predicted_iou']
                first_image = item['mask_file']

        if first_image:
            mask_path = os.path.join(auto_masks_dir, first_image)
            mask = Image.open(mask_path).convert('1')
            mask = np.array(mask)
            iou = calculate_iou(mask, ground_truth)
            iou_results.append(iou)

    mean_iou = np.mean(iou_results)
    print(f"SAM IOU Mean IoU: {mean_iou}")

    iou_results.clear()
    for image_id, items in images.items():
        ground_truth_path = os.path.join(
            ground_truth_dir, image_id + '_Segmentation.png')
        ground_truth = Image.open(ground_truth_path).convert('1')
        ground_truth = np.array(ground_truth)
        first_image = items[0]['mask_file']

        if first_image:
            mask_path = os.path.join(auto_masks_dir, first_image)
            mask = Image.open(mask_path).convert('1')
            mask = np.array(mask)
            iou = calculate_iou(mask, ground_truth)
            iou_results.append(iou)

    mean_iou1 = np.mean(iou_results)
    print(f"First Mask Mean IoU: {mean_iou1}")

    results_path = os.path.join(
        root_path, 'results/average_iou', 'sam_auto_best_iou.txt')
    with open(results_path, 'w') as f:
        f.write(f"Best SAM IOU Mean IoU: {mean_iou}\n")
        f.write(f"First Mask Mean IoU: {mean_iou1}\n")


if __name__ == '__main__':
    main()
