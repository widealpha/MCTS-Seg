import math
import os
import random
from typing import List, Optional, Set
import numpy as np
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from models.model import RewardPredictionModel
from data.mcts_loader import get_mcts_test_loader
from utils.helpers import get_checkpoints_path, get_mcts_path, setup_seed, load_sam, device
setup_seed()
sam = load_sam()
checkpoints_path = get_checkpoints_path()


class GlobalInfo:
    def __init__(self, predictor: SamPredictor, reward_model: RewardPredictionModel):
        # 初始化 SAM 预测器和 RewardModel
        self.predictor = predictor
        self.reward_model = reward_model
        # 设置 MCTS 参数
        # 预测max_points个点
        self.max_points = 3
        # 每次网格划分为K*K块
        self.grid_size = 4
        # 每次模拟的次数
        self.num_simulations = 50
        # 允许使用背景点
        self.enable_background = False

    def set_image(self, image: torch.Tensor):
        # (C, H, W)
        self.image = image
        # 将(C,H,W)转换为(H,W,C)
        self.image_array = image.permute(1, 2, 0).cpu().numpy()
        # 扩充为(1, H, W, C)
        self.single_image = image.unsqueeze(0)
        self.batch_size = 4
        # 扩充为(B, H, W, C)
        self.batch_image = image.unsqueeze(
            0).repeat(self.batch_size, 1, 1, 1).to(device)
        # 获取图像的宽和高
        self.width = image.shape[2]
        self.height = image.shape[1]
        predictor.set_image(self.image_array)
        # 网格划分的最大深度
        self.max_depth = int(
            math.log(min(self.width, self.height), self.grid_size))

    def __str__(self):
        return (f"GlobalInfo(max_points={self.max_points}, grid_size={self.grid_size}, "
                f"num_simulations={self.num_simulations}, enable_background={self.enable_background})")


class Action:
    def __init__(self, action: list = None, label: int = None):
        self.action = action
        self.label = label

    def __str__(self):
        return f"Action({self.action}, {self.label})"

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.action == other.action and self.label == other.label
        return False

    def __hash__(self):
        return hash((tuple(self.action), self.label))


class State:
    def __init__(self, global_info: GlobalInfo, taken_action: Optional[Set[Action]] = None, cur_action: Optional[Action] = None):
        if taken_action is None:
            taken_action = set()
        if cur_action is None:
            cur_action = Action()
        self.taken_action = taken_action
        self.cur_action = cur_action
        self.global_info = global_info
        self.reward = None

    def get_legal_actions(self):
        grid_size = self.global_info.grid_size
        actions: List[Action] = []

        # 初始节点
        if self.cur_action.action is None:
            for j in range(grid_size * grid_size):
                base_action = [j]
                actions.append(Action(base_action, 1))
                if global_info.enable_background:
                    actions.append(Action(base_action, 0))
            return actions
        # 超过最大深度
        if len(self.cur_action.action) >= self.global_info.max_depth:
            return []
        # 探索每一层的可能性
        for i in range(len(self.cur_action.action)):
            for j in range(grid_size * grid_size):
                # todo过滤之前遍历过的点
                base_action = self.cur_action.action[:i]
                if j != self.cur_action.action[i]:
                    base_action.append(j)
                    new_action = Action(base_action, 1)
                    if new_action not in self.taken_action:
                        actions.append(new_action)
                    if global_info.enable_background:
                        new_action = Action(base_action, 0)
                        if new_action not in self.taken_action:
                            actions.append(new_action)
        # 探索更下一层的可能性
        for j in range(grid_size * grid_size):
            base_action = self.cur_action.action.copy()
            base_action.append(j)
            new_action = Action(base_action, 1)
            if new_action not in self.taken_action:
                actions.append(new_action)
            if global_info.enable_background:
                new_action = Action(base_action, 0)
                if new_action not in self.taken_action:
                    actions.append(new_action)
        # for i in range(len(self.cur_action)):
        #     for j in range(self.grid_size ** 2):
        #         new_action = self.cur_action[:i]
        #         new_action.append(j)
        #         actions.append(new_action)
        # for j in range(self.grid_size ** 2):
        #     new_action = self.cur_action.copy()
        #     new_action.append(j)
        #     actions.append(new_action)
        # 排除 taken_action 中的部分
        # filtered_actions = []
        # for action in actions:
        #     if not any(set(action) == set(taken) for taken in self.taken_action):
        #         filtered_actions.append(action)
        # todo 根据规则排序actions
        return actions

    def take_action(self, action: Action):
        new_taken_action = self.taken_action.copy()
        new_taken_action.add(action)
        return State(taken_action=new_taken_action, cur_action=action, global_info=self.global_info)

    def action2point(self, action: Action):
        image_width = self.global_info.width
        image_height = self.global_info.height
        grid_size = self.global_info.grid_size

        x_start = 0
        y_start = 0
        current_w = image_width
        current_h = image_height

        for num in action.action:
            cell_w = current_w / grid_size
            cell_h = current_h / grid_size

            row = num // grid_size
            col = num % grid_size

            x_start += col * cell_w
            y_start += row * cell_h

            current_w = cell_w
            current_h = cell_h

        x_center = x_start + current_w / 2
        y_center = y_start + current_h / 2
        return (x_center, y_center)
        # base_x = 0
        # base_y = 0

        # for i in range(len(action.action)):
        #     base_x += image_width // grid_size * \
        #         (action.action[i] // grid_size)
        #     base_y += image_height // grid_size * \
        #         (action.action[i] % grid_size)
        #     image_width //= grid_size
        #     image_height //= grid_size
        # return (base_x + image_width // 2, base_y + image_height // 2)

    def all_action_points(self):
        points = []
        for action in self.taken_action:
            points.append(self.action2point(action))
        return points

    def all_action_labels(self):
        labels = []
        for action in self.taken_action:
            labels.append(action.label)
        return labels

    def get_reward(self):
        if (self.reward is not None):
            return self.reward
        points = np.array(self.all_action_points())
        labels = np.array(self.all_action_labels())
        # 使用 SAM 生成新的 mask
        new_masks, _, _ = self.global_info.predictor.predict(
            points, labels, multimask_output=False)
        # 使用 RewardModel 计算 reward
        mask = torch.Tensor(new_masks[0]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = self.global_info.reward_model(
                global_info.single_image, mask)
            self.reward = reward.item()
            return self.reward


class Node:
    def __init__(self, state: 'State', parent: 'Node' = None):
        self.state: State = state
        self.parent: Node = parent
        self.children: list[Node] = []
        self.visits = 0
        self.q_value = 0
        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_child(self, exploration_weight=1.414):
        scores = [
            (child.q_value / child.visits) +
            exploration_weight *
            math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[scores.index(max(scores))]

    def add_child(self, action,  child_state: 'State'):
        child_node = Node(state=child_state, parent=self)
        self.children.append(child_node)
        self.untried_actions.remove(action)
        return child_node

    def update(self, reward: float):
        self.q_value += reward
        self.visits += 1


class MCTS:
    def __init__(self, root: Node, global_info: GlobalInfo):
        self.root = root
        self.global_info = global_info

    def search(self, num_simulations) -> Node:
        for _ in tqdm(range(num_simulations), desc='MCTS', position=1, leave=False):
            node = self.select()
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.select_child(exploration_weight=0)

    def select(self):
        current = self.root
        while not current.is_fully_expanded() and current.children:
            current = current.select_child()
        # 如果有未扩展的动作则扩展
        if current.untried_actions:
            return self.expand(current)
        return current

    def expand(self, node: Node):
        """扩展阶段：创建新子节点"""
        # action = random.choice(node.untried_actions)
        # new_state = node.state.take_action(action)
        # return node.add_child(action, new_state)
        legal_actions = node.untried_actions
        max_reward = -np.inf
        for action in legal_actions:
            new_state = node.state.take_action(action)
            reward = self.get_reward(new_state)
            if reward > max_reward:
                max_reward = reward
                best_action = action
        return node.add_child(action, node.state.take_action(best_action))
        # all_points = []
        # all_labels = []
        # for action in legal_actions:
        #     new_state = node.state.take_action(action)
        #     all_points.append(new_state.all_action_points())
        #     all_labels.append(new_state.all_action_labels())
        # # 选取使得当前reward最大的Action
        # batch_reward_idx = self.get_batch_reward_idx(all_points, all_labels)

        # return node.add_child(node.state.take_action(legal_actions[batch_reward_idx]))

    def simulate(self, node: Node):
        current_state = node.state
        # while len(current_state.cur_action.action) < self.global_info.max_depth:
        #     legal_actions = current_state.get_legal_actions()
        #     # todo 替换随机策略
        #     action = legal_actions[np.random.randint(len(legal_actions))]
        #     current_state = current_state.take_action(action)
        return self.get_reward(current_state)

    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.update(reward)
            node = node.parent

    def get_reward(self, state: State):
        points = np.array(state.all_action_points())
        labels = np.array(state.all_action_labels())
        new_masks, _, _ = self.global_info.predictor.predict(
            points, labels, multimask_output=False)
        mask = torch.Tensor(new_masks[0]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = self.global_info.reward_model(
                self.global_info.single_image, mask)
        return reward.item()

    def get_batch_reward_idx(self, points, labels):
        batch_size = self.global_info.batch_size
        total_samples = len(points)
        num_to_pad = batch_size - total_samples % batch_size  # 计算需要补足的数量

        # 如果需要补足样本
        if num_to_pad > 0:
            last_point = points[-1]
            last_label = labels[-1]
            points.extend([last_point] * num_to_pad)
            labels.extend([last_label] * num_to_pad)
        torch_points = torch.Tensor(np.array(points)).to(device)
        torch_labels = torch.Tensor(np.array(labels)).to(device)
        batch_size = global_info.batch_size
        total_samples = torch_points.shape[0]
        rewards_list = []

        for i in range(0, len(points), batch_size):
            batch_points = torch_points[i:i + batch_size]
            batch_labels = torch_labels[i:i + batch_size]

            mask, *_ = self.global_info.predictor.predict_torch(
                batch_points, batch_labels, multimask_output=False
            )
            with torch.no_grad():
                rewards = global_info.reward_model(
                    self.global_info.batch_image, mask)
                rewards_list.append(rewards)

        # 合并所有小批次的奖励
        all_rewards = torch.cat(rewards_list, dim=0)
        # 返回所有奖励中最大的元素的索引
        return all_rewards.argmax().item()


def load_model(model_name):
    model_path = os.path.join(checkpoints_path, model_name)
    model = RewardPredictionModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # model = torch.load(model_path).to(device)
    model.eval()
    return model


def calculate_iou(mask1, mask2):
    """
    计算两个掩码之间的交并比IOU。
    :param mask1: 第一个掩码
    :param mask2: 第二个掩码
    :return: IOU 值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def sam_seg_cal_reward(predictor, points, labels, ground_truth, image_id):
    """
    使用 SAM 进行分割，结合 ground truth 计算 reward，reward 直接使用 IOU，结果保存在 results/mcts 文件夹下。
    :param predictor: SAM 预测器
    :param points: 分割点
    :param labels: 分割标签
    :param image: 输入图像(C, H, W)
    :param ground_truth: ground truth 掩码(H, W)
    :param image_id: 图像 ID
    """
    # 使用 SAM 生成新的 mask
    new_masks, _, _ = predictor.predict(points, labels, multimask_output=False)
    new_mask = new_masks[0]  # 取第一个 mask (H, W)

    # 计算 IOU 作为 reward
    iou = calculate_iou(new_mask, ground_truth)

    # 保存结果
    results_dir = get_mcts_path()
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f'{image_id}_result.png')
    mask_path = os.path.join(results_dir, f'{image_id}_mask.png')
    iou_path = os.path.join(results_dir, f'{image_id}_iou.txt')

    # 保存分割结果和 mask
    new_mask_image = (new_mask * 255).astype(np.uint8)
    ground_truth_image = (ground_truth * 255).astype(np.uint8)
    combined_image = np.concatenate(
        (new_mask_image, ground_truth_image), axis=1)
    Image.fromarray(combined_image).save(result_path)
    Image.fromarray(new_mask_image).save(mask_path)
    # 将 points 绘制在 mask 上并保存
    mask_with_points = Image.fromarray(ground_truth_image).convert("RGB")
    draw = ImageDraw.Draw(mask_with_points)
    radius = 6
    for idx, (point, label) in enumerate(zip(points, labels)):
        x, y = point
        # 根据标签值选择颜色
        color = 'green' if label == 0 else 'red'
        # 绘制圆形
        draw.ellipse((x - radius, y - radius, x + radius,
                     y + radius), outline=color, width=2)
        # 使用 textbbox 计算文本边界框大小
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=None)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        # 计算文本位置，确保文本居中
        text_x = x - text_size[0] // 2
        text_y = y - text_size[1] // 2

        # 绘制文本
        draw.text((text_x, text_y), text, fill=color)
    mask_with_points_path = os.path.join(
        results_dir, f'{image_id}_mask_with_points.png')
    mask_with_points.save(mask_with_points_path)
    # 保存 IOU
    with open(iou_path, 'w') as f:
        f.write(f'{iou}')

    return iou


# 示例用法
if __name__ == '__main__':
    test_loader = get_mcts_test_loader()
    # 初始化 SAM和RewardModel
    predictor = SamPredictor(sam)
    reward_model = load_model(model_name='latest.pth')
    global_info = GlobalInfo(predictor=predictor, reward_model=reward_model)
    results_dir = get_mcts_path()
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'info.log'), 'w') as f:
        f.write(f"Global Info: {global_info}\n")
    for data in tqdm(test_loader, desc='Test Image', position=0):
        image = data['image'][0].to(device)
        mask = data['mask'][0].to(device)
        global_info.set_image(image)
        initial_state = State(global_info=global_info)
        root = Node(initial_state)

        max_points = global_info.max_points
        best_node = root
        for _ in range(max_points):
            mcts = MCTS(best_node, global_info)
            best_node = mcts.search(
                num_simulations=global_info.num_simulations)
        points = np.array(best_node.state.all_action_points())
        labels = np.array(best_node.state.all_action_labels())

        reward = best_node.state.get_reward()
        image_id = data['image_id'][0]
        sam_seg_cal_reward(predictor=predictor, points=points,
                           labels=labels, ground_truth=mask[0].cpu().numpy(), image_id=image_id)
        # 将最佳点和奖励写入文件

        result_file_path = os.path.join(
            results_dir, f'{image_id}_best_points_and_reward.txt')
        with open(result_file_path, 'w') as f:
            f.write(f"Best points: {points.tolist()}\n")
            f.write(f"Best labels: {labels.tolist()}\n")
            f.write(f"Reward: {reward}\n")
    with open(os.path.join(results_dir, 'info.log'), 'a') as f:
        # 计算所有f'{image_id}_iou.txt'的均值并追加进去
        iou_results = []
        for file in os.listdir(results_dir):
            if file.endswith('_iou.txt'):
                with open(os.path.join(results_dir, file), 'r') as iou_f:
                    iou = float(iou_f.read())
                    iou_results.append(iou)
        mean_iou = np.mean(iou_results)
        f.write(f"Mean IoU: {mean_iou}\n")
    with open(os.path.join(results_dir, 'info.log'), 'a') as f:
        # 计算所有f'{image_id}_iou.txt'的均值并追加进去
        reward_results = []
        for file in os.listdir(results_dir):
            if file.endswith('_reward.txt'):
                with open(os.path.join(results_dir, file), 'r') as iou_f:
                    for line in file:
                        if line.startswith("Reward:"):
                            reward = float(line.split(":")[1].strip())
                            reward_results.append(reward)
        mean_reward = np.mean(reward_results)
        f.write(f"Mean Reward: {mean_reward}\n")

    print("Done!")
