import math
import os
import numpy as np
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from models.model import RewardPredictionModel
from data.mcts_loader import get_mcts_test_loader
from utils.helpers import get_checkpoints_path, get_log_path, get_mcts_path, setup_seed, load_sam, device
setup_seed()
sam = load_sam()
checkpoints_path = get_checkpoints_path()


class GlobalInfo:
    def __init__(self, image, predictor: SamPredictor, reward_model: RewardPredictionModel, image_shape=(1024, 1024)):
        self.image = image.permute(1, 2, 0).cpu().numpy()
        self.batch_image = image.unsqueeze(0)
        self.batch_size = 4
        self.batch_image_tensor = image.unsqueeze(
            0).repeat(self.batch_size, 1, 1, 1).to(device)
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.predictor = predictor
        self.reward_model = reward_model
        self.max_points = 5
        self.grid_size = 4
        self.max_depth = int(
            math.log(min(self.width, self.height), self.grid_size))
        predictor.set_image(self.image)


class State:
    def __init__(self, taken_action=[], action=[]):
        self.action = action
        self.taken_action = taken_action
        self.grid_size = global_info.grid_size
        self.reward = None

    def get_legal_actions(self):
        if len(self.action) >= global_info.max_depth:
            return []
        actions = []
        for i in range(len(self.action)):
            for j in range(self.grid_size ** 2):
                new_action = self.action[:i]
                new_action.append(j)
                actions.append(new_action)
        for j in range(self.grid_size ** 2):
            new_action = self.action.copy()
            new_action.append(j)
            actions.append(new_action)
        # 排除 taken_action 中的部分
        filtered_actions = []
        for action in actions:
            if not any(set(action) == set(taken) for taken in self.taken_action):
                filtered_actions.append(action)

        return filtered_actions

    def take_action(self, action):
        new_taken_action = self.taken_action.copy()
        new_taken_action.append(action)
        return State(taken_action=new_taken_action, action=action)

    def action2point(self, action, image_width: int, image_height: int):
        base_x = 0
        base_y = 0
        for i in range(len(action)):
            base_x += image_width // self.grid_size * \
                (action[i] // self.grid_size)
            base_y += image_height // self.grid_size * \
                (action[i] % self.grid_size)
            image_width //= self.grid_size
            image_height //= self.grid_size
        return (base_x + image_width // 2, base_y + image_height // 2)

    def all_points(self, global_info):
        image_width = global_info.width
        image_height = global_info.height
        points = []
        for action in self.taken_action:
            points.append(self.action2point(action, image_width, image_height))
        return points

    def get_reward(self, global_info: GlobalInfo):
        if (self.reward is not None):
            return self.reward
        points = np.array(self.all_points(global_info))
        labels = np.ones(len(points)).astype(int)
        # 使用 SAM 生成新的 mask
        new_masks, _, _ = global_info.predictor.predict(
            points, labels, multimask_output=False)
        # 使用 RewardModel 计算 reward
        mask = torch.Tensor(new_masks[0]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = global_info.reward_model(global_info.batch_image, mask)
            self.reward = reward.item()
            return self.reward


class Node:
    def __init__(self, state: 'State', parent: 'Node' = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        weights = [
            (child.reward / child.visits) + exploration_weight *
            np.sqrt(2 * np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(weights)]

    def add_child(self, child_state):
        child_node = Node(child_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.reward += reward
        self.visits += 1


class MCTS:
    def __init__(self, root: Node, global_info: GlobalInfo):
        self.root = root
        self.global_info = global_info

    def search(self, num_simulations) -> Node:
        # 这里添加tqdm
        for _ in tqdm(range(num_simulations), desc='MCTS', position=1):
            node = self.select(self.root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.best_child(exploration_weight=0)

    def select(self, node: Node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return self.expand(node)

    def expand(self, node: Node):
        legal_actions = node.state.get_legal_actions()
        all_points = []
        all_labels = []
        for action in legal_actions:
            new_state = node.state.take_action(action)
            all_points.append(new_state.all_points(self.global_info))
            all_labels.append(np.ones(len(all_points[-1])).astype(int))
        batch_reward_idx = self.get_batch_reward_idx(all_points, all_labels)

        return node.add_child(node.state.take_action(legal_actions[batch_reward_idx]))

    def simulate(self, node: Node):
        current_state = node.state
        while len(current_state.action) < 3:
            legal_actions = current_state.get_legal_actions()
            action = legal_actions[np.random.randint(len(legal_actions))]
            current_state = current_state.take_action(action)
        return current_state.get_reward(self.global_info)

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

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
                    self.global_info.batch_image_tensor, mask)
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
    for idx, point in enumerate(points):
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius,
                     y + radius), outline='red', width=2)

        # 使用 textbbox 计算文本边界框大小
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=None)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        text_x = x - text_size[0] // 2
        text_y = y - text_size[1] // 2
        draw.text((text_x, text_y), text, fill='red')
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
    predictor = SamPredictor(sam)  # 初始化 SAM
    for data in tqdm(test_loader, desc='Test Image', position=0, leave=True):
        image = data['image'][0].to(device)
        mask = data['mask'][0].to(device)

        reward_model = load_model(model_name='2025-02-24_20-49-30.pth')
        # 初始化 RewardModel
        global_info = GlobalInfo(
            image=image, predictor=predictor, reward_model=reward_model, image_shape=image.shape[1:])
        initial_state = State()
        root = Node(initial_state)

        max_points = global_info.max_points
        best_node = root
        for _ in range(max_points):
            mcts = MCTS(best_node, global_info)
            best_node = mcts.search(num_simulations=40)
        points = np.array(best_node.state.all_points(global_info))
        reward = best_node.state.get_reward(global_info)
        labels = np.ones(len(points)).astype(int)
        image_id = data['image_id'][0]
        sam_seg_cal_reward(predictor=predictor, points=points,
                           labels=labels, ground_truth=mask[0].cpu().numpy(), image_id=image_id,)
        # 将最佳点和奖励写入文件
        results_dir = os.path.join('results', 'mcts')
        os.makedirs(results_dir, exist_ok=True)
        result_file_path = os.path.join(
            results_dir, f'{image_id}_best_points_and_reward.txt')
        with open(result_file_path, 'w') as f:
            f.write(f"Best points: {points.tolist()}\n")
            f.write(f"Reward: {reward}\n")
