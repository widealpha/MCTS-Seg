import os
import random
import re
import time
import cv2
import numpy as np
import torch
from PIL import Image
from gym.vector.utils import spaces
from segment_anything import sam_model_registry, SamPredictor

import math
import random

from models.model import RewardPredictionModel
from utils.helpers import setup_seed, load_sam, device, calculate_iou, get_log_writer
setup_seed()
sam = load_sam()

class Node:
    def __init__(self, state, parent=None):
        # 自身的
        self.depth = 0
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child_node = Node(child_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda child: child.wins / child.visits + exploration_weight * math.sqrt(
            math.log(self.visits) / child.visits))


class MCTS:
    def __init__(self, iterations, predictor):
        self.iterations = iterations
        self.predictor = predictor

    def search(self, root):
        for _ in range(self.iterations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return root.best_child(exploration_weight=0)

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return self.expand(node)

    def expand(self, node):
        legal_actions = node.state.get_legal_actions()
        best_action = None
        best_reward = float('-inf')

        # 遍历所有合法动作，使用 SAM 计算每个动作的 reward
        for action in legal_actions:
            # 获取执行当前动作后生成的状态
            new_state = node.state.take_action(action)
            reward = new_state.get_reward()
            # 更新最高 reward 和对应的最佳动作
            if reward > best_reward:
                best_reward = reward
                best_action = action
        if best_action is not None:
            return node.add_child(node.state.take_action(best_action))
        return node

    def simulate(self, node):
        current_state = node.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.take_action(action)
        return current_state.get_reward()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent


class GameState:
    def __init__(self, predictor: SamPredictor, sam_encoder, image, reward_model, points=None, max_step=None):
        self.image = image  # 读取图像
        self.reward_model = reward_model  # 读取真实标签

        if points:
            self.points = points
        else:
            self.points = self.divide_image_into_grid(image_shape=self.image.shape, grid_size=(20, 20))  # 将图像分成网格
            self.actions = list(range(len(self.points) * 2))
            self.action_space = spaces.Discrete(len(self.actions))
        self.max_step = max_step
        self.current_step = 0
        self.visited = []
        self.predictor = predictor
        self.sam_encoder = sam_encoder
        self.reward = None

    def copy(self):
        new_state = GameState(predictor=self.predictor, sam_encoder=self.sam_encoder, image=self.image,
                              reward_model=self.reward_model,
                              points=self.points, max_step=self.max_step)
        new_state.actions = self.actions
        new_state.action_space = self.action_space
        new_state.visited = self.visited.copy()
        new_state.current_step = self.current_step
        return new_state

    def divide_image_into_grid(self, image_shape, grid_size):
        """将图像划分为网格并返回中心点坐标"""
        height, width = image_shape[:2]
        rows, cols = grid_size
        cell_width = width // cols
        cell_height = height // rows
        centers = [(j * cell_width + cell_width // 2, i * cell_height + cell_height // 2)
                   for i in range(rows) for j in range(cols)]
        return centers

    def get_legal_actions(self):
        return list(set(self.actions) - set(self.visited))

    def take_action(self, action):
        """执行动作并生成新状态"""
        # 检查动作是否有效
        if action not in self.get_legal_actions():
            raise ValueError(f"Action {action} is not legal.")
        new_state = self.copy()
        new_state.visited.append(action)
        # 更新步骤计数
        new_state.current_step += 1
        new_state.reward = new_state.get_reward()
        # 返回新状态（可以根据需要返回状态、奖励等）
        return new_state

    def apply_sam(self):
        """应用SAM模型在指定位置进行分割"""
        points = []
        labels = []
        for action in self.visited:
            point, label = self.action_2_point_label(action)
            points.append(point)
            labels.append(label)
        coords_torch = torch.as_tensor(points, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        masks, _, _ = self.predictor.predict_torch(point_coords=coords_torch, point_labels=labels_torch,
                                                   multimask_output=False)
        return masks[0]

    def is_terminal(self):
        return self.current_step >= self.max_step

    def get_reward(self):
        """使用IOU作为奖励"""
        if self.reward:
            return self.reward
        mask = self.apply_sam()
        mask_tensor = mask.float()
        # 添加一个新维度并沿最后一个维度重复3次
        mask_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        # mask_predictor: SamPredictor = self.mask_predictor
        # mask_predictor.set_image(mask)
        image_tensor = torch.tensor(self.image).float().permute(2, 0, 1).unsqueeze(0).to(device)
        # print(image_tensor.shape)
        # print(mask_tensor.shape)
        with torch.no_grad():
            # mask_feature = self.sam_encoder(mask_tensor)
            # image_feature = self.predictor.get_image_embedding()
            # return self.reward_model(image_feature, mask_feature)
            return self.reward_model(image_tensor, mask_tensor)

    def action_2_point_label(self, action):
        return self.points[action // 2], (action + 1) % 2

    def get_point_labels(self):
        points = []
        labels = []
        for action in self.visited:
            # 获取对应的点和标签
            point, label = self.action_2_point_label(action)
            points.append(point)
            labels.append(label)
        return points, labels

    def get_info(self):
        res = f'current_step: {self.current_step}\n'
        for action in self.visited:
            # 获取对应的点和标签
            point, label = self.action_2_point_label(action)
            res += f'Action: {action}, Point: {point}, Label: {label}\n'
        return res


def get_real_iou(sam_predict, points, labels, ground_truth):
    mask = sam_predict.predict(np.array(points), np.array(labels), multimask_output=False)
    mask = mask[0]
    mask = cv2.resize(mask, ground_truth.shape)
    return calculate_iou(mask, ground_truth)


def train(sam, reward_model, image, ground_truth, max_step=2, iterations=4):
    log_writer = get_log_writer()
    predictor = SamPredictor(sam)
    sam_encoder = sam.image_encoder
    # image = cv2.imread(image_path)  # 读取图像
    # ground_truth_mask = cv2.imread(ground_truth_mask, 0)  # 读取真实标签
    predictor.set_image(image)
    initial_state = GameState(image=image, reward_model=reward_model, predictor=predictor,
                              sam_encoder=sam_encoder,
                              max_step=max_step)
    root_node = Node(initial_state)
    start_time = time.time()
    mcts = MCTS(iterations=iterations, predictor=predictor)
    best_action: Node = mcts.search(root_node)
    while not best_action.state.is_terminal():
        best_action = mcts.search(best_action)
    # 获取奖励信息
    reward = best_action.state.get_reward()
    end_time = time.time()
    elapsed_time = end_time - start_time
    points, labels = best_action.state.get_point_labels()
    mask, _, __ = predictor.predict(np.array(points), np.array(labels), multimask_output=False)
    resized_mask = (mask[0] * 255).astype(np.uint8)
    resized_mask = cv2.resize(resized_mask, (ground_truth.shape[1], ground_truth.shape[0]))
    image = torch.tensor(image)
    resized_mask = torch.tensor(resized_mask)
    ground_truth = torch.tensor(ground_truth)
    iou = calculate_iou(resized_mask, ground_truth)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Best action details: {best_action.state.get_info()}")
    print(f"Best action reward: {reward}")
    print(f"IoU: {iou}")
    log_writer.add_image('Result/origin', image, dataformats='HWC')
    log_writer.add_image('Result/mask', resized_mask, dataformats='HW')
    log_writer.add_image('Result/ground_truth', ground_truth, dataformats='HW')
    log_writer.add_text('Result/action', f'Points: {points}, Labels: {labels}')
    log_writer.add_scalar('Result/reward', reward)
    log_writer.add_scalar('Result/IoU', iou)
    log_writer.close()

def load_model():
    model = RewardPredictionModel().to(device)
    with open('reward_model/checkpoint/latest') as f:
        model_path = f'reward_model/checkpoint/{f.read()}'  # 模型权重文件路径
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_dataset():
    dataset = []
    image_dir = "data/processed/ISBI2016_ISIC/test"
    file_list = os.listdir(image_dir)
    image_files = [file.split('.')[0] for file in file_list if re.match(r'^ISIC_\d{7}\.png$', file)]
    image_files.sort()
    for idx, image_id in enumerate(image_files):
        dataset.append((idx, f'{os.path.join(image_dir, f"{image_id}.png")}',
                        f'{os.path.join(image_dir, f"{image_id}_mask_0.png")}'))
    return dataset


def main():
    setup_seed(2024)
    model = load_model()
    sam = load_sam()
    for data in load_dataset():
        idx, image_path, ground_truth_path = data
        print(f'正在处理{data}')
        image = cv2.imread(image_path)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        train(sam, model, image, ground_truth, max_step=4, iterations=4)


def test_model():
    setup_seed(2024)
    model = RewardPredictionModel().to(device)
    model_path = 'reward_model/checkpoint/2024-11-15_13-41-07.pth'  # 模型权重文件路径
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image_path = 'data/processed/ISBI2016_ISIC/test/ISIC_0000000.png'
    image = cv2.imread(image_path)
    mask_path = 'data/processed/ISBI2016_ISIC/train/ISIC_0000000_mask_5.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = torch.tensor(mask).float().to(device)
    gray_image_rgb = torch.stack([mask] * 3, dim=0).unsqueeze(0)
    sam = load_sam()
    sam_predictor = SamPredictor(sam)
    sam_predictor.set_image(image)
    image_feature = sam_predictor.get_image_embedding()
    image_encoder = sam.image_encoder
    with torch.no_grad():
        mask_feature = image_encoder(gray_image_rgb)
        reward = model(image_feature, mask_feature)
        print(reward)


if __name__ == '__main__':
    main()
    # test_model()
    # image_id = 'ISIC_0000000'
    # max_step = 1
    # train(image_path=f'data/ISIC-2017_Training_Data/{image_id}.jpg',
    #       ground_truth_mask=f'data/ISIC-2017_Training_Part1_GroundTruth/{image_id}_segmentation.png',
    #       log_path=f'test/{image_id}_{max_step}.txt',
    #       max_step=max_step)
