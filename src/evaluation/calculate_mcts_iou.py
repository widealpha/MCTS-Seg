import os
import re
from collections import defaultdict

from utils.helpers import get_root_path
root_path = get_root_path()


def read_iou_files(results_dir):
    iou_files = [f for f in os.listdir(
        results_dir) if re.match(r'.*_iou\.txt$', f)]
    iou_values = []

    for iou_file in iou_files:
        iou_file_path = os.path.join(results_dir, iou_file)

        with open(iou_file_path, 'r') as f:
            iou_value = float(f.read().replace('IOU:', '').strip())
            iou_values.append(iou_value)

    return iou_values


def calculate_average_iou(iou_values):
    if not iou_values:
        return 0
    return sum(iou_values) / len(iou_values)


def save_average_iou(average_iou, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, 'mcts_average_iou.txt')

    with open(result_file, 'w') as f:
        f.write(f"average_iou: {average_iou}\n")


if __name__ == '__main__':
    results_dir = os.path.join(root_path, 'results/mcts-3point')
    output_dir = os.path.join(root_path, 'results/average_iou')

    iou_values = read_iou_files(results_dir)
    average_iou = calculate_average_iou(iou_values)
    save_average_iou(average_iou, output_dir)
