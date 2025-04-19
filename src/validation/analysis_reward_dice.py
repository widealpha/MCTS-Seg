from src.utils.helpers import get_data_path, get_result_path, set_chinese_font
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import os
import re
import numpy as np

set_chinese_font()


def extract_dice_and_reward(file_path):
    """从文件中提取 Dice 和 Reward 值"""
    with open(file_path, 'r') as f:
        content = f.read()
    dice_match = re.search(r"Dice:\s*([\d.]+)", content)
    reward_match = re.search(r"Reward:\s*([\d.]+)", content)
    if dice_match and reward_match:
        dice = float(dice_match.group(1))
        reward = float(reward_match.group(1))
        return dice, reward
    return None, None


def analyze_and_plot(input_dir):
    """分析 Dice 和 Reward 的相关性并绘制散点图"""
    dice_values = []
    reward_values = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_result.txt"):
            file_path = os.path.join(input_dir, file_name)
            dice, reward = extract_dice_and_reward(file_path)
            if dice is not None and reward is not None:
                dice_values.append(dice)
                reward_values.append(reward)

    if dice_values and reward_values:
        # 计算皮尔逊相关系数
        correlation, _ = pearsonr(dice_values, reward_values)
        print(f"Dice 和 Reward 的皮尔逊相关系数: {correlation:.4f}")

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(dice_values, reward_values,
                    color='blue', alpha=0.7, label='数据点')

        # 添加趋势线
        z = np.polyfit(dice_values, reward_values, 1)
        p = np.poly1d(z)
        plt.plot(dice_values, p(dice_values),
                 color='red', linestyle='--', label='趋势线')

        # 图形美化
        plt.title("Dice 和 Reward 的关系", fontsize=14)
        plt.xlabel("Dice", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 显示图形
        plt.show()
    else:
        print("未找到有效的 Dice 和 Reward 数据。")


if __name__ == "__main__":
    input_dir = os.path.join(get_result_path(), 'mcts', 'ISIC2016')
    analyze_and_plot(input_dir)
