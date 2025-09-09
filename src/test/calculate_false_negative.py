import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import torch
import torch.nn.functional as F
from src.utils.helpers import calculate_iou, calculate_dice

def load_mask(mask_path):
    """加载掩码图像并转换为二值图像"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法加载掩码: {mask_path}")
    # 将掩码转换为二值图像 (0, 255) -> (0, 1)
    mask = (mask > 127).astype(np.uint8)
    return mask

def calculate_hausdorff_distance(pred_mask, gt_mask, use_gpu=True):
    """计算Hausdorff Distance和HD95，支持GPU加速"""
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    # 获取边界点
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)
    
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return float('inf'), float('inf')
    
    # 如果边界点很少，使用CPU计算避免GPU开销
    if len(pred_boundary) < 100 and len(gt_boundary) < 100:
        use_gpu = False
        device = torch.device('cpu')
    
    if use_gpu and device.type == 'cuda':
        # GPU加速版本
        hd, hd95 = calculate_hausdorff_gpu(pred_boundary, gt_boundary, device)
    else:
        # CPU版本（原始实现）
        hd1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        hd2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        hd = max(hd1, hd2)
        
        # 计算HD95
        distances1 = [min([np.linalg.norm(p - q) for q in gt_boundary]) for p in pred_boundary]
        distances2 = [min([np.linalg.norm(q - p) for p in pred_boundary]) for q in gt_boundary]
        all_distances = distances1 + distances2
        
        if len(all_distances) > 0:
            hd95 = np.percentile(all_distances, 95)
        else:
            hd95 = float('inf')
    
    return hd, hd95

def calculate_hausdorff_gpu(pred_boundary, gt_boundary, device):
    """GPU加速的Hausdorff Distance计算"""
    try:
        # 转换为PyTorch张量
        pred_tensor = torch.tensor(pred_boundary, dtype=torch.float32, device=device)
        gt_tensor = torch.tensor(gt_boundary, dtype=torch.float32, device=device)
        
        # 计算距离矩阵（批量计算以节省内存）
        batch_size = 1000  # 批处理大小
        
        # 计算从pred到gt的最小距离
        min_distances_pred_to_gt = []
        for i in range(0, len(pred_tensor), batch_size):
            batch_pred = pred_tensor[i:i+batch_size]
            # 计算距离矩阵：[batch_size, gt_size]
            distances = torch.cdist(batch_pred.unsqueeze(0), gt_tensor.unsqueeze(0)).squeeze(0)
            min_dist = torch.min(distances, dim=1)[0]
            min_distances_pred_to_gt.append(min_dist)
        
        min_distances_pred_to_gt = torch.cat(min_distances_pred_to_gt)
        
        # 计算从gt到pred的最小距离
        min_distances_gt_to_pred = []
        for i in range(0, len(gt_tensor), batch_size):
            batch_gt = gt_tensor[i:i+batch_size]
            distances = torch.cdist(batch_gt.unsqueeze(0), pred_tensor.unsqueeze(0)).squeeze(0)
            min_dist = torch.min(distances, dim=1)[0]
            min_distances_gt_to_pred.append(min_dist)
        
        min_distances_gt_to_pred = torch.cat(min_distances_gt_to_pred)
        
        # 计算HD（最大距离）
        hd1 = torch.max(min_distances_pred_to_gt).item()
        hd2 = torch.max(min_distances_gt_to_pred).item()
        hd = max(hd1, hd2)
        
        # 计算HD95
        all_min_distances = torch.cat([min_distances_pred_to_gt, min_distances_gt_to_pred])
        hd95 = torch.quantile(all_min_distances, 0.95).item()
        
        return hd, hd95
        
    except torch.cuda.OutOfMemoryError:
        print("GPU内存不足，回退到CPU计算")
        # 回退到CPU计算
        hd1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        hd2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        hd = max(hd1, hd2)
        
        distances1 = [min([np.linalg.norm(p - q) for q in gt_boundary]) for p in pred_boundary]
        distances2 = [min([np.linalg.norm(q - p) for p in pred_boundary]) for q in gt_boundary]
        all_distances = distances1 + distances2
        
        if len(all_distances) > 0:
            hd95 = np.percentile(all_distances, 95)
        else:
            hd95 = float('inf')
            
        return hd, hd95

def get_boundary_points(mask):
    """提取掩码的边界点，使用优化的方法"""
    if mask.sum() == 0:  # 空掩码
        return np.array([])
    
    # 使用形态学操作快速获取边界
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded
    
    # 获取边界点坐标
    boundary_coords = np.where(boundary > 0)
    if len(boundary_coords[0]) == 0:
        return np.array([])
    
    # 返回(x, y)格式的坐标点
    boundary_points = np.column_stack((boundary_coords[1], boundary_coords[0]))
    
    # 如果边界点太多，进行下采样以提高速度
    if len(boundary_points) > 5000:
        step = len(boundary_points) // 5000
        boundary_points = boundary_points[::step]
    
    return boundary_points

def calculate_confusion_metrics(pred_mask, gt_mask):
    """计算混淆矩阵指标"""
    # 展平数组以便计算
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    # 使用utils中的函数计算IoU和Dice
    iou = calculate_iou(pred_mask, gt_mask, threshold=0.5)
    dice = calculate_dice(pred_mask, gt_mask, threshold=0.5)
    
    # 计算HD和HD95
    hd, hd95 = calculate_hausdorff_distance(pred_mask, gt_mask)
    
    return {
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'iou': iou,
        'dice': dice,
        'hd': hd,
        'hd95': hd95
    }

def calculate_metrics_for_directory(pred_dir, gt_dir, is_mcts=False):
    """计算目录中所有图像的混淆矩阵指标"""
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    
    if not pred_path.exists():
        raise ValueError(f"预测目录不存在: {pred_dir}")
    if not gt_path.exists():
        raise ValueError(f"GT目录不存在: {gt_dir}")
    
    # 获取所有图像文件
    if is_mcts:
        # 对于MCTS目录，只获取以_mask.png结尾的文件
        pred_files = list(pred_path.glob("*_mask.png"))
    else:
        # 对于baseline目录，获取所有图像文件
        pred_files = list(pred_path.glob("*.png")) + list(pred_path.glob("*.jpg"))
        if not pred_files:
            pred_files = list(pred_path.glob("*.bmp")) + list(pred_path.glob("*.tif"))
    
    total_files = len(pred_files)
    print(f"找到 {total_files} 个预测文件")
    
    results = []
    total_metrics = {'true_negative': 0, 'false_positive': 0, 'false_negative': 0, 'true_positive': 0}
    total_iou = 0
    total_dice = 0
    total_hd = 0
    total_hd95 = 0
    fn_values = []  # 存储每张图片的FN值用于计算平均值
    individual_metrics = []  # 存储每张图片的各种指标用于计算平均值
    
    for file_idx, pred_file in enumerate(pred_files):
        # 显示进度
        if (file_idx + 1) % 50 == 0 or file_idx == 0:
            print(f"处理进度: {file_idx + 1}/{total_files} ({(file_idx + 1)/total_files*100:.1f}%)")
        if is_mcts:
            # 对于MCTS文件，从{image_id}_mask.png提取image_id，找对应的{image_id}.png GT文件
            image_id = pred_file.stem.replace('_mask', '')
            gt_file = gt_path / f"{image_id}.png"
            
            # 如果.png不存在，尝试其他扩展名
            if not gt_file.exists():
                for ext in ['.jpg', '.bmp', '.tif']:
                    gt_file = gt_path / f"{image_id}{ext}"
                    if gt_file.exists():
                        break
        else:
            # 对于baseline文件，直接找同名GT文件
            gt_file = gt_path / pred_file.name
            if not gt_file.exists():
                # 尝试其他可能的扩展名
                stem = pred_file.stem
                for ext in ['.png', '.jpg', '.bmp', '.tif']:
                    gt_file = gt_path / f"{stem}{ext}"
                    if gt_file.exists():
                        break
        
        if not gt_file.exists():
            if is_mcts:
                print(f"警告: 找不到对应的GT文件: {image_id}.png (预测文件: {pred_file.name})")
            else:
                print(f"警告: 找不到对应的GT文件: {pred_file.name}")
            continue
        
        try:
            # 加载掩码
            pred_mask = load_mask(pred_file)
            gt_mask = load_mask(gt_file)
            
            # 确保掩码尺寸相同
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            
            # 计算混淆矩阵指标
            metrics = calculate_confusion_metrics(pred_mask, gt_mask)
            
            # 添加文件名信息
            if is_mcts:
                metrics['filename'] = pred_file.name
                metrics['gt_filename'] = gt_file.name
                metrics['image_id'] = image_id
            else:
                metrics['filename'] = pred_file.name
            
            results.append(metrics)
            
            # 累加总计
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # 累加IoU和Dice
            total_iou += metrics['iou']
            total_dice += metrics['dice']
            
            # 累加HD和HD95 (跳过无限值)
            if not np.isinf(metrics['hd']):
                total_hd += metrics['hd']
            if not np.isinf(metrics['hd95']):
                total_hd95 += metrics['hd95']
            
            # 存储FN值
            fn_values.append(metrics['false_negative'])
            
            # 计算每张图片的各种指标
            tn, fp, fn, tp = metrics['true_negative'], metrics['false_positive'], metrics['false_negative'], metrics['true_positive']
            img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            img_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            img_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            img_f1_score = 2 * (img_precision * img_recall) / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0
            img_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            individual_metrics.append({
                'precision': img_precision,
                'recall': img_recall,
                'specificity': img_specificity,
                'f1_score': img_f1_score,
                'accuracy': img_accuracy,
                'iou': metrics['iou'],
                'dice': metrics['dice'],
                'hd': metrics['hd'] if not np.isinf(metrics['hd']) else 0,
                'hd95': metrics['hd95'] if not np.isinf(metrics['hd95']) else 0
            })
                
        except Exception as e:
            print(f"处理文件 {pred_file.name} 时出错: {e}")
    
    # 计算平均值
    num_images = len(results)
    avg_iou = total_iou / num_images if num_images > 0 else 0
    avg_dice = total_dice / num_images if num_images > 0 else 0
    avg_fn = np.mean(fn_values) if fn_values else 0
    
    # 计算有效HD值的数量（非无穷大）
    valid_hd_count = sum(1 for result in results if not np.isinf(result['hd']))
    valid_hd95_count = sum(1 for result in results if not np.isinf(result['hd95']))
    
    avg_hd = total_hd / valid_hd_count if valid_hd_count > 0 else float('inf')
    avg_hd95 = total_hd95 / valid_hd95_count if valid_hd95_count > 0 else float('inf')
    
    # 计算图片级平均指标
    avg_metrics = {}
    if individual_metrics:
        for metric in ['precision', 'recall', 'specificity', 'f1_score', 'accuracy', 'iou', 'dice', 'hd', 'hd95']:
            avg_metrics[f'avg_{metric}'] = np.mean([m[metric] for m in individual_metrics])
    
    return results, total_metrics, avg_iou, avg_dice, avg_fn, avg_metrics, avg_hd, avg_hd95

def calculate_derived_metrics(metrics):
    """计算派生指标"""
    tn, fp, fn, tp = metrics['true_negative'], metrics['false_positive'], metrics['false_negative'], metrics['true_positive']
    
    # 避免除零错误
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'iou': iou
    }

def main():
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU不可用，将使用CPU计算")
    
    # 定义路径
    baseline_dir = "/home/kmh/ai/MCTS-Seg/results/baseline/ISIC2016/medical_sam_adapter/center_1"
    mcts_dir = "/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2016/m2g6s512a1bg0t0rt0f1/msa-2025-06-29_14-24-03"
    # mcts_dir = "/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2016/m2g8s512a1bg0t0rt0f1/msa-2025-06-29_03-58-09"
    
    
    # baseline_dir = "/home/kmh/ai/MCTS-Seg/results/baseline/ISIC2016/medical_sam_adapter/center_1"
    # mcts_dir = "/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2016/m2g6s512a1bg0t0rt0f1/msa-2025-06-29_14-24-03"
    
    gt_dir = "/home/kmh/ai/MCTS-Seg/data/ISIC2016/raw/test/gt"
    
    print("计算混淆矩阵指标...")
    print("=" * 60)
    
    # 计算baseline结果
    print("\n1. Baseline (Medical SAM Adapter) 结果:")
    try:
        baseline_results, baseline_total, baseline_avg_iou, baseline_avg_dice, baseline_avg_fn, baseline_avg_metrics, baseline_avg_hd, baseline_avg_hd95 = calculate_metrics_for_directory(baseline_dir, gt_dir, is_mcts=False)
        baseline_derived = calculate_derived_metrics(baseline_total)
        
        print(f"处理的图像数量: {len(baseline_results)}")
        print("\n混淆矩阵:")
        print(f"  True Negative:  {baseline_total['true_negative']:,}")
        print(f"  False Positive: {baseline_total['false_positive']:,}")
        print(f"  False Negative: {baseline_total['false_negative']:,}")
        print(f"  True Positive:  {baseline_total['true_positive']:,}")
        
        print("\n像素级累加指标:")
        print(f"  Precision:   {baseline_derived['precision']:.4f}")
        print(f"  Recall:      {baseline_derived['recall']:.4f}")
        print(f"  Specificity: {baseline_derived['specificity']:.4f}")
        print(f"  F1-Score:    {baseline_derived['f1_score']:.4f}")
        print(f"  Accuracy:    {baseline_derived['accuracy']:.4f}")
        print(f"  IoU:         {baseline_derived['iou']:.4f}")
        
        print("\n图片级平均指标:")
        if baseline_avg_metrics:
            print(f"  平均Precision:   {baseline_avg_metrics['avg_precision']:.4f}")
            print(f"  平均Recall:      {baseline_avg_metrics['avg_recall']:.4f}")
            print(f"  平均Specificity: {baseline_avg_metrics['avg_specificity']:.4f}")
            print(f"  平均F1-Score:    {baseline_avg_metrics['avg_f1_score']:.4f}")
            print(f"  平均Accuracy:    {baseline_avg_metrics['avg_accuracy']:.4f}")
            print(f"  平均IoU:         {baseline_avg_metrics['avg_iou']:.4f}")
            print(f"  平均Dice:        {baseline_avg_metrics['avg_dice']:.4f}")
            print(f"  平均HD:          {baseline_avg_metrics['avg_hd']:.4f}")
            print(f"  平均HD95:        {baseline_avg_metrics['avg_hd95']:.4f}")
        print(f"  平均FN:          {baseline_avg_fn:.2f}")
        
    except Exception as e:
        print(f"处理baseline结果时出错: {e}")
    
    # 计算MCTS结果
    print("\n" + "=" * 60)
    print("\n2. MCTS 结果:")
    try:
        mcts_results, mcts_total, mcts_avg_iou, mcts_avg_dice, mcts_avg_fn, mcts_avg_metrics, mcts_avg_hd, mcts_avg_hd95 = calculate_metrics_for_directory(mcts_dir, gt_dir, is_mcts=True)
        mcts_derived = calculate_derived_metrics(mcts_total)
        
        print(f"处理的图像数量: {len(mcts_results)}")
        print("\n混淆矩阵:")
        print(f"  True Negative:  {mcts_total['true_negative']:,}")
        print(f"  False Positive: {mcts_total['false_positive']:,}")
        print(f"  False Negative: {mcts_total['false_negative']:,}")
        print(f"  True Positive:  {mcts_total['true_positive']:,}")
        
        print("\n像素级累加指标:")
        print(f"  Precision:   {mcts_derived['precision']:.4f}")
        print(f"  Recall:      {mcts_derived['recall']:.4f}")
        print(f"  Specificity: {mcts_derived['specificity']:.4f}")
        print(f"  F1-Score:    {mcts_derived['f1_score']:.4f}")
        print(f"  Accuracy:    {mcts_derived['accuracy']:.4f}")
        print(f"  IoU:         {mcts_derived['iou']:.4f}")
        
        print("\n图片级平均指标:")
        if mcts_avg_metrics:
            print(f"  平均Precision:   {mcts_avg_metrics['avg_precision']:.4f}")
            print(f"  平均Recall:      {mcts_avg_metrics['avg_recall']:.4f}")
            print(f"  平均Specificity: {mcts_avg_metrics['avg_specificity']:.4f}")
            print(f"  平均F1-Score:    {mcts_avg_metrics['avg_f1_score']:.4f}")
            print(f"  平均Accuracy:    {mcts_avg_metrics['avg_accuracy']:.4f}")
            print(f"  平均IoU:         {mcts_avg_metrics['avg_iou']:.4f}")
            print(f"  平均Dice:        {mcts_avg_metrics['avg_dice']:.4f}")
            print(f"  平均HD:          {mcts_avg_metrics['avg_hd']:.4f}")
            print(f"  平均HD95:        {mcts_avg_metrics['avg_hd95']:.4f}")
        print(f"  平均FN:          {mcts_avg_fn:.2f}")
        
    except Exception as e:
        print(f"处理MCTS结果时出错: {e}")
    
    # 保存详细结果到CSV
    if 'baseline_results' in locals() and baseline_results:
        df_baseline = pd.DataFrame(baseline_results)
        df_baseline.to_csv('baseline_confusion_metrics.csv', index=False)
        print(f"\nBaseline详细结果已保存到: baseline_confusion_metrics.csv")
    
    if 'mcts_results' in locals() and mcts_results:
        df_mcts = pd.DataFrame(mcts_results)
        df_mcts.to_csv('mcts_confusion_metrics.csv', index=False)
        print(f"MCTS详细结果已保存到: mcts_confusion_metrics.csv")
    
    # 比较结果
    if 'baseline_derived' in locals() and 'mcts_derived' in locals():
        print("\n" + "=" * 60)
        print("\n3. 方法比较:")
        print("\n像素级累加指标比较:")
        print(f"{'指标':<12} {'Baseline':<10} {'MCTS':<10} {'提升':<10}")
        print("-" * 45)
        for metric in ['precision', 'recall', 'f1_score', 'accuracy', 'iou']:
            baseline_val = baseline_derived[metric]
            mcts_val = mcts_derived[metric]
            improvement = mcts_val - baseline_val
            print(f"{metric.capitalize():<12} {baseline_val:<10.4f} {mcts_val:<10.4f} {improvement:+.4f}")
        
        if 'baseline_avg_metrics' in locals() and 'mcts_avg_metrics' in locals() and baseline_avg_metrics and mcts_avg_metrics:
            print("\n图片级平均指标比较:")
            print(f"{'指标':<12} {'Baseline':<10} {'MCTS':<10} {'提升':<10}")
            print("-" * 45)
            for metric in ['precision', 'recall', 'f1_score', 'accuracy', 'iou', 'dice', 'hd', 'hd95']:
                baseline_val = baseline_avg_metrics[f'avg_{metric}']
                mcts_val = mcts_avg_metrics[f'avg_{metric}']
                # 对于HD和HD95，数值越小越好，所以计算减少量
                if metric in ['hd', 'hd95']:
                    improvement = baseline_val - mcts_val  # 正值表示改进
                    print(f"{metric.upper():<12} {baseline_val:<10.4f} {mcts_val:<10.4f} {improvement:+.4f}")
                else:
                    improvement = mcts_val - baseline_val
                    print(f"{metric.capitalize():<12} {baseline_val:<10.4f} {mcts_val:<10.4f} {improvement:+.4f}")
    
    # 分析IoU最差的30%数据的提升
    if 'baseline_results' in locals() and 'mcts_results' in locals() and baseline_results and mcts_results:
        print("\n" + "=" * 60)
        print("\n4. IoU最差30%数据的提升分析:")
        
        # 创建baseline的IoU排序索引
        baseline_iou_list = [(i, result['iou']) for i, result in enumerate(baseline_results)]
        baseline_iou_list.sort(key=lambda x: x[1])  # 按IoU从小到大排序
        
        # 计算30%的数量
        worst_30_percent_count = int(len(baseline_iou_list) * 0.3)
        worst_indices = [idx for idx, _ in baseline_iou_list[:worst_30_percent_count]]
        
        print(f"最差30%数据数量: {worst_30_percent_count}")
        print(f"IoU范围: {baseline_iou_list[0][1]:.4f} - {baseline_iou_list[worst_30_percent_count-1][1]:.4f}")
        
        # 匹配MCTS结果并计算提升
        matched_improvements = []
        baseline_worst_30_metrics = []
        mcts_worst_30_metrics = []
        
        for idx in worst_indices:
            baseline_result = baseline_results[idx]
            baseline_filename = baseline_result['filename']
            
            # 在MCTS结果中找到对应的文件
            mcts_result = None
            if 'image_id' in baseline_result:
                # 如果baseline也有image_id（适用于MCTS格式）
                target_id = baseline_result['image_id']
                for mcts_res in mcts_results:
                    if 'image_id' in mcts_res and mcts_res['image_id'] == target_id:
                        mcts_result = mcts_res
                        break
            else:
                # 直接按文件名匹配
                for mcts_res in mcts_results:
                    if mcts_res['filename'] == baseline_filename:
                        mcts_result = mcts_res
                        break
                    # 如果MCTS是_mask.png格式，尝试匹配原文件名
                    if 'image_id' in mcts_res:
                        if f"{mcts_res['image_id']}.png" == baseline_filename:
                            mcts_result = mcts_res
                            break
            
            if mcts_result:
                # 计算各项指标的提升
                improvements = {}
                for metric in ['iou', 'dice', 'hd', 'hd95']:
                    baseline_val = baseline_result[metric]
                    mcts_val = mcts_result[metric]
                    # 对于HD和HD95，数值越小越好
                    if metric in ['hd', 'hd95']:
                        improvements[metric] = baseline_val - mcts_val  # 正值表示改进
                    else:
                        improvements[metric] = mcts_val - baseline_val
                
                # 计算混淆矩阵指标的提升
                for prefix in ['baseline', 'mcts']:
                    result = baseline_result if prefix == 'baseline' else mcts_result
                    tn, fp, fn, tp = result['true_negative'], result['false_positive'], result['false_negative'], result['true_positive']
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    if prefix == 'baseline':
                        baseline_precision, baseline_recall, baseline_f1 = precision, recall, f1_score
                    else:
                        improvements['precision'] = precision - baseline_precision
                        improvements['recall'] = recall - baseline_recall
                        improvements['f1_score'] = f1_score - baseline_f1
                
                matched_improvements.append(improvements)
                baseline_worst_30_metrics.append(baseline_result)
                mcts_worst_30_metrics.append(mcts_result)
        
        if matched_improvements:
            print(f"成功匹配的数据数量: {len(matched_improvements)}")
            
            # 计算平均提升
            avg_improvements = {}
            for metric in ['iou', 'dice', 'precision', 'recall', 'f1_score', 'hd', 'hd95']:
                improvements_list = [imp[metric] for imp in matched_improvements]
                avg_improvements[metric] = np.mean(improvements_list)
                
            print(f"\n最差30%数据的平均提升:")
            print(f"{'指标':<12} {'平均提升':<10}")
            print("-" * 25)
            for metric in ['iou', 'dice', 'precision', 'recall', 'f1_score', 'hd', 'hd95']:
                print(f"{metric.upper() if metric in ['hd', 'hd95'] else metric.capitalize():<12} {avg_improvements[metric]:+.4f}")
            
            # 计算baseline最差30%的平均指标
            baseline_worst_30_avg = {}
            mcts_worst_30_avg = {}
            
            for metric in ['iou', 'dice', 'hd', 'hd95']:
                baseline_worst_30_avg[metric] = np.mean([result[metric] for result in baseline_worst_30_metrics])
                mcts_worst_30_avg[metric] = np.mean([result[metric] for result in mcts_worst_30_metrics])
            
            print(f"\n最差30%数据的绝对值比较:")
            print(f"{'指标':<12} {'Baseline':<10} {'MCTS':<10} {'提升':<10}")
            print("-" * 45)
            for metric in ['iou', 'dice', 'hd', 'hd95']:
                baseline_val = baseline_worst_30_avg[metric]
                mcts_val = mcts_worst_30_avg[metric]
                if metric in ['hd', 'hd95']:
                    improvement = baseline_val - mcts_val  # 正值表示改进
                    print(f"{metric.upper():<12} {baseline_val:<10.4f} {mcts_val:<10.4f} {improvement:+.4f}")
                else:
                    improvement = mcts_val - baseline_val
                    print(f"{metric.capitalize():<12} {baseline_val:<10.4f} {mcts_val:<10.4f} {improvement:+.4f}")
            
            # 统计提升的数据比例
            positive_improvements = {metric: sum(1 for imp in matched_improvements if imp[metric] > 0) for metric in ['iou', 'dice', 'precision', 'recall', 'f1_score', 'hd', 'hd95']}
            
            print(f"\n提升的数据比例:")
            print(f"{'指标':<12} {'提升数量':<10} {'总数量':<10} {'比例':<10}")
            print("-" * 45)
            for metric in ['iou', 'dice', 'precision', 'recall', 'f1_score', 'hd', 'hd95']:
                count = positive_improvements[metric]
                total = len(matched_improvements)
                ratio = count / total if total > 0 else 0
                print(f"{metric.upper() if metric in ['hd', 'hd95'] else metric.capitalize():<12} {count:<10} {total:<10} {ratio:<10.2%}")
        else:
            print("未找到匹配的MCTS结果数据")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()