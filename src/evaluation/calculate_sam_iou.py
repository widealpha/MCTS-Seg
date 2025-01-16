import os
import re
from collections import defaultdict

from utils.helpers import get_root_path
root_path = get_root_path()


def read_iou_files(resize_dir):
    iou_files = [f for f in os.listdir(
        resize_dir) if re.match(r'.*_mask_\d+_iou\.txt$', f)]
    iou_data = defaultdict(list)

    for iou_file in iou_files:
        mask_id = re.search(r'_mask_(\d+)_iou\.txt$', iou_file).group(1)
        iou_file_path = os.path.join(resize_dir, iou_file)

        with open(iou_file_path, 'r') as f:
            iou_value = float(f.read().strip())
            iou_data[mask_id].append(iou_value)

    return iou_data


def calculate_average_iou(iou_data):
    average_iou = {}
    for mask_id, iou_values in iou_data.items():
        average_iou[mask_id] = sum(iou_values) / \
            len(iou_values) if iou_values else 0
    return average_iou


def save_average_iou(average_iou, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, 'average_iou_results.txt')

    with open(result_file, 'w') as f:
        for mask_id, avg_iou in average_iou.items():
            f.write(f"sam_mask_{mask_id}: {avg_iou}\n")


if __name__ == '__main__':

    resize_dir = os.path.join(root_path, 'data/processed/test/resized')
    results_dir = os.path.join(root_path, 'results/average_iou')

    iou_data = read_iou_files(resize_dir)
    average_iou = calculate_average_iou(iou_data)
    save_average_iou(average_iou, results_dir)
