# 统计分析与可视化脚本
# 用于分析 detailed_results.json 并输出最佳方案、优势数值、比例和可视化图
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot(result_dir):
    result_path = os.path.join(result_dir, 'detailed_results.json')
    if not os.path.exists(result_path):
        print(f"未找到结果文件: {result_path}")
        return
    with open(result_path, 'r') as f:
        all_results = json.load(f)
    # 统计每张图片4种case的dice_mean
    img_cases = {}
    for r in all_results:
        img_id = r['image_id']
        case = f"{r['point_type']}_label{r['label']}"
        if img_id not in img_cases:
            img_cases[img_id] = {}
        img_cases[img_id][case] = r['dice_mean']
    cases = sorted(list({case for r in all_results for case in [f"{r['point_type']}_label{r['label']}"]}))
    # 统计每种case的平均dice
    case_means = {c: [] for c in cases}
    for img, d in img_cases.items():
        for c in cases:
            if c in d and d[c] is not None:
                case_means[c].append(d[c])
    case_avg = {c: np.mean(case_means[c]) for c in cases}
    # 找到最优case
    best_case = max(case_avg, key=case_avg.get)
    print(f"最优方案: {best_case}, 平均DICE={case_avg[best_case]:.4f}")
    for c in cases:
        if c != best_case:
            diff = case_avg[best_case] - case_avg[c]
            print(f"比 {c} 平均高 {diff:.4f}")
    # 统计每张图片最优case
    best_count = 0
    total = 0
    for img, d in img_cases.items():
        best = max(d, key=d.get)
        if best == best_case:
            best_count += 1
        total += 1
    print(f"最优方案在 {best_count}/{total} ({best_count/total:.2%}) 图片上表现最好")
    # 绘图
    plt.figure(figsize=(8,5))
    data = [case_means[c] for c in cases]
    plt.boxplot(data, labels=cases)
    plt.title('DICE分数分布（不同点类型和标签）')
    plt.ylabel('DICE')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(result_dir, 'dice_case_boxplot.png'))
    plt.show()

if __name__ == '__main__':
    # 修改为你的结果目录
    result_dir = './results/test/ISIC2016/all/medical_sam_adapter/random_1'  # 或实际目录
    analyze_and_plot(result_dir)
