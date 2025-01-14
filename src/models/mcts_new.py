import os
import numpy as np
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor
import torch
from models.model import RewardPredictionModel
from data.loader import get_data_loader
from utils.helpers import get_root_path, setup_seed, load_sam, device
setup_seed()
sam = load_sam()
root_path = get_root_path()


class GlobalInfo:
    def __init__(self, image, predictor: SamPredictor, reward_model: RewardPredictionModel, image_shape=(1024, 1024)):
        self.image = image
        self.batch_image = image.unsqueeze(0)
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.predictor = predictor
        self.reward_model = reward_model


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
        new_taken_action.append(self.action)
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

    def all_points(self, image_width: int, image_height: int):
        points = [self.action2point(self.action, image_width, image_height)]
        for action in self.taken_action:
            points.append(self.action2point(action, image_width, image_height))
        return points

    def get_reward(self, global_info: GlobalInfo):
        points = np.array(self.all_points(
            global_info.width, global_info.height))
        labels = np.ones(len(points)).astype(int)
        # print(points, labels)
        # 使用 SAM 生成新的 mask
        new_masks, _, _ = global_info.predictor.predict(
            points, labels, multimask_output=False)
        print(new_masks[0].shape)
        # 使用 RewardModel 计算 reward
        with torch.no_grad():
            reward = global_info.reward_model(
                global_info.batch_image, new_masks[0])
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

    def search(self, num_simulations):
        for _ in range(num_simulations):
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
        while current_state.depth < 3:
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


# 示例用法
if __name__ == '__main__':
    train_loader, test_loader = get_data_loader(batch_size=1)
    for data in test_loader:
        image = data['image'][0].to(device)
        initial_state = State()
        root = Node(initial_state)
        predictor = SamPredictor(sam)  # 初始化 SAM
        predictor.set_image(image)
        model_path = os.path.join(
            root_path, 'results/models/2025-01-13_23-48-35.pth')
        reward_model = load_model()
        # 初始化 RewardModel
        global_info = GlobalInfo(
            image=image, predictor=predictor, reward_model=reward_model, image_shape=image.shape)
        mcts = MCTS(root, global_info)
        best_node = mcts.search(num_simulations=1000)
        print(
            f"Best point: {best_node.state.point}, Reward: {best_node.reward / best_node.visits}")
        break
