import os
import numpy as np

def calculate_metrics(iou_value):
    """
    Calculate mIoU and mDice from IoU values.
    """
    return iou_value, 2 * iou_value / (iou_value + 1)

def process_files(baseline_dir):
    """
    Process files in baseline_dir and result_dir to calculate mIoU and mDice.
    """
    
    all_data = []
    for file_name in os.listdir(baseline_dir):
        if file_name.endswith("_score.txt"):
            baseline_path = os.path.join(baseline_dir, file_name)

            # Read IoU values from baseline file
            with open(baseline_path, 'r') as f:
                iou_value = np.array([float(line.strip()) for line in f])
                all_data.append((iou_value, file_name))
    # Sort all_data by IoU values
    all_data.sort(key=lambda x: np.mean(x[0]))

    # Split into first half and second half
    mid_index = len(all_data) // 2
    first_half = all_data[:mid_index]
    second_half = all_data[mid_index:]
    return first_half, second_half
    
def calculate_half_data(first_half, second_half, result_dir):
    # Calculate metrics for baseline_dir combined with first_half and second_half
    baseline_first_metrics = [
        calculate_metrics(iou_values) for iou_values, _ in first_half
    ]
    baseline_first_results = np.mean(baseline_first_metrics, axis=0)
    baseline_second_metrics = [
        calculate_metrics(iou_values) for iou_values, _ in second_half
    ]
    baseline_second_results = np.mean(baseline_second_metrics, axis=0)

    # Calculate metrics for result_dir combined with first_half and second_half
    result_first_metrics = []
    result_second_metrics = []

    for _, file_name in first_half:
        result_path = os.path.join(result_dir, file_name.replace("_score.txt", "_iou.txt"))
        # Check if the result file exists
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_iou = np.array([float(line.strip()) for line in f])
                result_first_metrics.append(calculate_metrics(result_iou))

    for _, file_name in second_half:
        result_path = os.path.join(result_dir, file_name.replace("_score.txt", "_iou.txt"))
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_iou = np.array([float(line.strip()) for line in f])
                result_second_metrics.append(calculate_metrics(result_iou))
                
    result_first_results = np.mean(result_first_metrics, axis=0)
    result_second_results = np.mean(result_second_metrics, axis=0)
    print("Baseline First Half Metrics:")
    print(f"mIoU: {baseline_first_results[0]}, mDice: {baseline_first_results[1]}")

    print("\nBaseline Second Half Metrics:")
    print(f"mIoU: {baseline_second_results[0]}, mDice: {baseline_second_results[1]}")

    print("\nResult First Half Metrics:")
    print(f"mIoU: {result_first_results[0]}, mDice: {result_first_results[1]}")

    print("\nResult Second Half Metrics:")
    print(f"mIoU: {result_second_results[0]}, mDice: {result_second_results[1]}")
    

if __name__ == "__main__":
    baseline_dir = "/home/kmh/mcts/data/ISIC2016/processed/test/baseline/auto"
    result_dir = "/home/kmh/mcts/result/mcts/ISIC2016-v1"
    first_half, second_half = process_files(baseline_dir)
    calculate_half_data(first_half, second_half, result_dir)