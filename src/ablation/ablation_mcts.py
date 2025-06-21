from tqdm import tqdm
from src.models.mcts import DynamicGrid, sam_seg_cal_reward, get_sam_predictor, load_model, predict, calculate_iou_dice
from src.data.mcts_loader import get_mcts_test_loader
from src.utils.helpers import get_device, get_ablation_result_path, setup_seed
import numpy as np
import os
import torch


def traverse_grid_and_segment(predictor, reward_model, image, mask, grid, image_id, output_dir):
    """
    使用 DynamicGrid 遍历，记录最佳点的顺序，并进行分割。
    """
    points_path = []
    labels_path = []
    current_selection = []

    while not grid.is_fully_subdivided(current_selection):
        # 获取当前层的所有可能动作
        actions = grid.get_possible_actions(current_selection)["children"]
        best_reward = -float('inf')
        best_action = None

        # 遍历所有子动作，找到最佳 reward 的点
        for action, (x, y) in actions:
            points_array = np.array([[x, y]])
            labels_array = np.array([1])  # 假设标签为前景点
            with torch.no_grad():
                _, rewards = predict(
                    predictor=predictor,
                    reward_model=reward_model,
                    points=points_array,
                    labels=labels_array,
                    image=image.unsqueeze(0),
                )
                reward = rewards.item()
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

        # 更新当前选择路径，并记录点和标签
        current_selection = best_action
        x, y = grid.get_coordinate(current_selection)
        points_path.append([x, y])
        labels_path.append(1)  # 假设标签为前景点

    # 使用最终的点进行分割
    points_array = np.array([points_path[-1]])
    labels_array = np.array([labels_path[-1]])
    iou, dice, reward = sam_seg_cal_reward(
        predictor=predictor,
        reward_model=reward_model,
        points=points_array,
        labels=labels_array,
        image=image,
        ground_truth=mask,
        image_id=image_id,
        output_dir=output_dir
    )

    # 返回结果
    return points_path, labels_path, iou, dice, reward


def main():
    setup_seed()
    device = get_device()
    predictor = get_sam_predictor()
    reward_model = load_model("latest.pth", 512, 512)  # 加载奖励模型

    dataloader = get_mcts_test_loader(batch_size=1, shuffle=False)
    output_dir = os.path.join(get_ablation_result_path(), 'wo_mcts')  # 保存结果的目录
    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc='Test Image', position=0):
        image = batch['image'][0].to(device)
        mask = batch['mask'][0].to(device)
        image_id = batch['image_id'][0]
        predictor.set_image(image.permute(1, 2, 0).cpu().numpy())
        grid = DynamicGrid(image.shape[1], 8)

        # 遍历网格并分割
        points, labels, iou, dice, reward = traverse_grid_and_segment(
            predictor, reward_model, image, mask, grid, image_id, output_dir
        )

        # 保存点的顺序和分割结果
        result_file = os.path.join(output_dir, f"{image_id}_result.txt")
        with open(result_file, "w") as f:
            f.write(f"Points: {[points[-1]]}, Path: {points}\n")
            f.write(f"Labels: {[labels[-1]]}, Path: {labels}\n")
            f.write(f"IoU: {iou}\n")
            f.write(f"Dice: {dice}\n")
            f.write(f"Reward: {reward}\n")
    calculate_iou_dice(results_dir=output_dir)


if __name__ == "__main__":
    main()
