import os
import numpy as np
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image
from tqdm import tqdm
from models.model import RewardPredictionModel
from data.loader import get_data_loader, get_mcts_test_loader
from utils.helpers import get_root_path, setup_seed, load_sam, device
setup_seed()
sam = load_sam()
root_path = get_root_path()


class GlobalInfo:
    def __init__(self, image, predictor: SamPredictor, reward_model: RewardPredictionModel, image_shape=(1024, 1024)):
        self.image = image.permute(1, 2, 0).cpu().numpy()
        self.batch_image = image.unsqueeze(0)
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.predictor = predictor
        self.reward_model = reward_model
        predictor.set_image(self.image)


class State:
    def __init__(self, taken_action=[], action=[], grid_size: int = 8):
        self.action = action
        self.taken_action = taken_action
        self.grid_size = grid_size

    def get_legal_actions(self):
        if len(self.action) >= 3:
            return []
        actions = []
        for i in range(self.grid_size ** 2):
            new_action = self.action.copy()
            new_action.append(i)
            actions.append(new_action)
        return actions

    def take_action(self, action):
        new_taken_action = self.taken_action.copy()
        new_taken_action.append(action)
        return State(taken_action=new_taken_action, action=action, grid_size=self.grid_size)

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
        points = np.array(self.all_points(global_info))
        labels = np.ones(len(points)).astype(int)
        # 使用 SAM 生成新的 mask
        new_masks, _, _ = global_info.predictor.predict(
            points, labels, multimask_output=False)
        # 使用 RewardModel 计算 reward
        mask = torch.Tensor(new_masks[0]).unsqueeze(
            0).repeat(3, 1, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = global_info.reward_model(global_info.batch_image, mask)
            return reward.item()


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
    def __init__(self, root: 'Node', global_info: 'GlobalInfo'):
        self.root = root
        self.global_info = global_info

    def search(self, num_simulations) -> Node:
        # 这里添加tqdm
        for _ in tqdm(range(num_simulations), desc='MCTS', position=1):
            node = self.select(self.root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.best_child(exploration_weight=0)

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return self.expand(node)

    def expand(self, node: Node):
        legal_actions = node.state.get_legal_actions()
        best_action = None
        best_reward = float('-inf')

        for action in legal_actions:
            new_state = node.state.take_action(action)
            reward = new_state.get_reward(self.global_info)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        if best_action is not None:
            return node.add_child(node.state.take_action(best_action))
        return node

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


def load_model():
    model_path = os.path.join(
        root_path, 'results/models/2025-01-13_23-48-35.pth')
    model = RewardPredictionModel().to(device)
    # with open(model_path) as f:
    #     model_path = f'reward_model/checkpoint/{f.read()}'  # 模型权重文件路径
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
    :param image: 输入图像
    :param ground_truth: ground truth 掩码
    :param image_id: 图像 ID
    """
    # 使用 SAM 生成新的 mask
    new_masks, _, _ = predictor.predict(points, labels, multimask_output=False)
    new_mask = new_masks[0]

    # 计算 IOU 作为 reward
    iou = calculate_iou(new_mask, ground_truth)

    # 保存结果
    results_dir = os.path.join(root_path, 'results', 'mcts')
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

    # 保存 IOU
    with open(iou_path, 'w') as f:
        f.write(f'IOU: {iou}\n')

    return iou


# 示例用法
if __name__ == '__main__':
    test_loader = get_mcts_test_loader()
    predictor = SamPredictor(sam)  # 初始化 SAM
    for data in tqdm(test_loader, desc='Test Image', position=0):
        image = data['image'][0].to(device)
        initial_state = State()
        root = Node(initial_state)
        model_path = os.path.join(
            root_path, 'results/models/2025-01-13_23-48-35.pth')
        reward_model = load_model()
        # 初始化 RewardModel
        global_info = GlobalInfo(
            image=image, predictor=predictor, reward_model=reward_model, image_shape=image.shape[1:])
        mcts = MCTS(root, global_info)
        best_node = mcts.search(num_simulations=20)
        points = np.array(best_node.state.all_points(global_info))
        reward = best_node.state.get_reward(global_info)
        labels = np.ones(len(points)).astype(int)
        image_id = data['image_id'][0]
        ground_truth_path = os.path.join(
            root_path, 'data/processed/test/resized', f"{image_id}_mask_0.png")
        ground_truth = Image.open(ground_truth_path).convert('L')
        sam_seg_cal_reward(predictor=predictor, points=points,
                           labels=labels, ground_truth=np.array(ground_truth), image_id=image_id,)
        # 将最佳点和奖励写入文件
        results_dir = os.path.join('results', 'mcts')
        os.makedirs(results_dir, exist_ok=True)
        result_file_path = os.path.join(results_dir, f'{image_id}_best_points_and_reward.txt')
        with open(result_file_path, 'w') as f:
            f.write(f"Best points: {points.tolist()}\n")
            f.write(f"Reward: {reward}\n")