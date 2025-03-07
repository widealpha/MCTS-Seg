from __future__ import annotations
import math
import os
import random
from typing import List, Optional, Any, Tuple, Dict
import numpy as np
from collections import defaultdict
from segment_anything import SamPredictor
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from models.model import RewardPredictionModel
from data.mcts_loader import get_mcts_test_loader
from utils.helpers import get_checkpoints_path, get_mcts_path, setup_seed, load_sam, device
setup_seed()
sam = load_sam()
checkpoints_path = get_checkpoints_path()
utils: Utils = None


class Utils:
    def __init__(self, predictor: SamPredictor, reward_model: RewardPredictionModel):
        # 初始化 SAM 预测器和 RewardModel
        self.predictor = predictor
        self.reward_model = reward_model
        # 设置 MCTS 参数
        # 预测max_points个点
        self.max_points = 1
        # 每次网格划分为K*K块
        self.grid_size = 4
        # 每次模拟的次数
        self.num_simulations = 200
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
        self.grid = DynamicGrid(
            n=min(self.width, self.height), m=self.grid_size)

    def __str__(self):
        return (f"GlobalInfo(max_points={self.max_points}, grid_size={self.grid_size}, "
                f"num_simulations={self.num_simulations}, enable_background={self.enable_background})")


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


def load_model(model_name, sample_width, sample_height):
    model_path = os.path.join(checkpoints_path, model_name)
    model = RewardPredictionModel(
        sample_width=sample_width, sample_height=sample_height).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


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


class DynamicGrid:
    """
    动态网格类：
    - n: 整个网格的尺寸（假设为 n*n 的正方形,n 为整数）
    - m: 每次划分时，将当前区域划分为 m*m 块
    """

    def __init__(self, n: int, m: int = 4) -> None:
        self.n = n
        self.m = m

    def _get_region(self, selection: List[int]) -> Tuple[int, int, float]:
        """
        根据选择列表计算当前区域的左上角坐标和区域尺寸
        返回 (top_left_x, top_left_y, size)
        """
        top_left_x: float = 0.0
        top_left_y: float = 0.0
        size: float = float(self.n)
        for sel in selection:
            index: int = sel - 1  # 1-indexed 转 0-indexed
            row: int = index // self.m
            col: int = index % self.m
            cell_size: float = size / self.m
            top_left_x += col * cell_size
            top_left_y += row * cell_size
            size = cell_size
        return int(round(top_left_x)), int(round(top_left_y)), size

    def get_coordinate(self, selection: List[int]) -> Tuple[int, int]:
        """
        根据选择列表返回当前区域中心点的坐标（整数）
        """
        top_left_x, top_left_y, size = self._get_region(selection)
        center_x: int = int(round(top_left_x + size / 2))
        center_y: int = int(round(top_left_y + size / 2))
        return center_x, center_y

    def get_possible_actions(self, selection: List[int]) -> Dict[str, List[Tuple[List[int], Tuple[int, int]]]]:
        """
        返回可能的动作，包括：
          - children: 对当前选中区域进行 m*m 细分后，每个子块的中心点，
                      新的选择路径为 selection + [子块编号]
          - siblings: 在父区域中，同一级别下除当前选中块之外的其他块，
                      新的选择路径为 selection[:-1] + [其他块编号]

        返回的字典键：
          "children": List[(new_selection, (x, y))]
          "siblings": List[(new_selection, (x, y))]
        """
        actions: Dict[str, List[Tuple[List[int], Tuple[int, int]]]] = {
            "siblings": [], "children": []}

        # 当前区域
        cur_top_left_x, cur_top_left_y, cur_size = self._get_region(selection)
        # 子动作：对当前区域进一步划分
        for i in range(1, self.m * self.m + 1):
            new_selection = selection + [i]
            index = i - 1
            row = index // self.m
            col = index % self.m
            cell_size = cur_size / self.m
            center_x = int(round(cur_top_left_x + col *
                           cell_size + cell_size / 2))
            center_y = int(round(cur_top_left_y + row *
                           cell_size + cell_size / 2))
            actions["children"].append((new_selection, (center_x, center_y)))

        # 兄弟动作：同级替换（需存在父区域，即 selection 非空）
        if selection:
            parent_top_left_x, parent_top_left_y, parent_size = self._get_region(
                selection[:-1])
            current_choice = selection[-1]
            for i in range(1, self.m * self.m + 1):
                if i == current_choice:
                    continue  # 跳过当前选择
                new_selection = selection[:-1] + [i]
                index = i - 1
                row = index // self.m
                col = index % self.m
                cell_size = parent_size / self.m
                center_x = int(round(parent_top_left_x +
                               col * cell_size + cell_size / 2))
                center_y = int(round(parent_top_left_y +
                               row * cell_size + cell_size / 2))
                actions["siblings"].append(
                    (new_selection, (center_x, center_y)))
        else:
            # 若 selection 为空，则兄弟动作与子动作相同
            actions["siblings"] = actions["children"].copy()

        return actions

    def is_fully_subdivided(self, selection: List[int]) -> bool:
        """
        判断当前区域是否已经划分到不能再划分的地步。
        判定标准：若对当前区域进一步划分(即 size / m)的结果小于 1,则认为无法再划分。

        参数:
            selection: 选择列表，表示当前划分路径

        返回:
            True 表示已不能再进一步划分，否则 False。
        """
        _, _, size = self._get_region(selection)
        cell_size = size / self.m
        return cell_size < 1

    @staticmethod
    def _test():
        # 假设整个网格大小为 100x100，每次划分为 3x3 块
        grid = DynamicGrid(n=512, m=4)

        # 设定一个选择路径，例如 [1, 2] 表示第一层选择1号块，第二层在该块中选择2号块
        selection: List[int] = [1, 2, 3, 4]

        # 获取当前区域中心点坐标
        coord: Tuple[int, int] = grid.get_coordinate(selection)
        print("当前选中点的坐标为:", coord)

        # 获取可能的动作（子动作和兄弟动作）
        actions = grid.get_possible_actions(selection)
        print("\n子动作 (进一步划分当前区域):")
        for new_sel, pos in actions["children"]:
            print(f"选择 {new_sel} => 坐标 {pos}")

        print("\n兄弟动作 (同级替换选择):")
        for new_sel, pos in actions["siblings"]:
            print(f"选择 {new_sel} => 坐标 {pos}")

        # 判断是否已经划分到不能再细分的地步
        if grid.is_fully_subdivided(selection):
            print("\n当前区域已不能再划分。")
        else:
            print("\n当前区域仍可进一步划分。")


class GameState:
    """
    定义游戏状态，保存历史采取的 action 以及当前的 action。
    - action_history: 历史中每一步的选择（列表的列表）
    - current_action: 当前状态的选择路径（列表），例如 [1, 2] 表示第一层选择 1 号块，第二层选择 2 号块
    - n: 整个网格的尺寸（n×n 的正方形，n 为整数）
    - m: 每次划分时的块数（每次将当前区域划分为 m×m 块）
    """

    def __init__(
        self,
        action_history: List[List[int]] = None,
        current_action: List[int] = None,
    ) -> None:
        self.action_history: List[List[int]
                                  ] = action_history if action_history is not None else []
        self.current_action: List[int] = current_action if current_action is not None else [
        ]
        self.reward: float = None

    def is_terminal(self) -> bool:
        """
        判断当前状态是否为终局。
        终局的判定依据：使用当前的 current_action 对应的区域，
        若该区域继续划分后每个子块尺寸小于 1，则认为无法再细分，即为终局。
        """
        return utils.grid.is_fully_subdivided(self.current_action)

    def get_possible_moves(self) -> List[List[int]]:
        """
        返回当前状态下所有合法的下一步动作（新的选择路径）。
        包括：
          - 当前区域进一步细分后的各子块（children）
          - 若当前状态非根状态，还包括同级替换（siblings）的选择
        """
        actions = utils.grid.get_possible_actions(self.current_action)
        moves: List[List[int]] = [move for move, coord in actions["children"]]
        if self.current_action:
            moves.extend([move for move, coord in actions["siblings"]])
        return moves
    
    def get_possible_children_moves(self) -> List[List[int]]:
        """
        返回当前状态下所有合法的下一步动作（新的选择路径）。
        包括：
          - 当前区域进一步细分后的各子块（children）
        """
        actions = utils.grid.get_possible_actions(self.current_action)
        moves: List[List[int]] = [move for move, coord in actions["children"]]
        return moves

    def apply_move(self, move: List[int]) -> GameState:
        """
        根据给定的动作（新的选择路径）生成新的状态。
        更新方式：将当前状态的 current_action 追加到 action_history 中，
        并将新的 move 作为新的 current_action 返回。
        """
        new_history = self.action_history.copy()
        if self.current_action:  # 记录历史（根状态 current_action 为空时不记录）
            new_history.append(self.current_action)
        return GameState(action_history=new_history, current_action=move)

    def all_action_points(self) -> List[Tuple[int, int]]:
        """
        根据历史动作和当前动作，计算所有对应的坐标点。
        这里利用 grid(DynamicGrid 实例)计算每个 action 的中心坐标。
        """
        all_actions = self.action_history.copy()
        if self.current_action:
            all_actions.append(self.current_action)
        points = [utils.grid.get_coordinate(action) for action in all_actions]
        return points

    def all_action_labels(self) -> List[int]:
        """
        根据历史动作和当前动作，生成对应的标签列表。
        这里简单假设所有点都是正样本，标签均为 1。
        """
        all_actions = self.action_history.copy()
        if self.current_action:
            all_actions.append(self.current_action)
        return [1 for _ in all_actions]

    def get_reward(self) -> float:
        """
        计算 reward：
          1. 根据 all_action_points() 与 all_action_labels() 获取所有点与标签，
          2. 使用 SAM 预测器生成新的 mask，
          3. 利用 RewardModel 根据 single_image 和生成的 mask 计算 reward，
          4. 将计算结果保存到 self.reward 并返回。
        """
        if self.reward is not None:
            return self.reward

        points = np.array(self.all_action_points())
        labels = np.array(self.all_action_labels())
        # 使用 SAM 生成新的 mask
        new_masks, _, _ = utils.predictor.predict(
            points, labels, multimask_output=False
        )
        # 使用 RewardModel 计算 reward
        mask = torch.Tensor(new_masks[0]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = utils.reward_model(utils.single_image, mask)
            self.reward = reward.item()
            return self.reward


class Node:
    """
    MCTS 中的树节点，每个节点保存一个状态，并记录探索过程中的统计信息。
    """

    def __init__(self, state: GameState, parent: Optional[Node] = None) -> None:
        self.state: GameState = state
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self.wins: float = 0.0
        self.visits: int = 0
        # 节点初始化时获得当前状态的所有可用动作
        self.untried_moves: List[Any] = state.get_possible_moves()

    def expand(self) -> Node:
        """
        从未尝试的动作中选择一个扩展子节点
        """
        move = self.untried_moves.pop()
        next_state = self.state.apply_move(move)
        child_node = Node(state=next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param: float = 1.4) -> Node:
        """
        使用 UCT(上置信界)公式选择最佳子节点
        """
        choices_weights = [
            (child.wins / child.visits) + c_param *
            math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def update(self, reward: float) -> None:
        """
        更新节点的访问次数和累计奖励
        """
        self.visits += 1
        self.wins += reward


class MCTS:
    """
    MCTS 算法封装类，通过多次迭代来搜索最优决策。
    """

    def __init__(self, root_state: GameState) -> None:
        self.root: Node = Node(state=root_state)

    def select(self, node: Node) -> Node:
        """
        选择阶段：从当前节点开始，沿着树不断选择最佳子节点，
        直到遇到未完全扩展的节点或终局状态。
        """
        while not node.state.is_terminal():
            if node.untried_moves:
                return node.expand()
            else:
                node = node.best_child()
        return node

    def simulation(self, state: GameState) -> float:
        """
        模拟阶段：从给定状态开始随机模拟直到终局，并返回模拟的奖励值。
        """
        current_state = state
        while not current_state.is_terminal():
            possible_moves = current_state.get_possible_children_moves()
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            current_state = current_state.apply_move(move)
        return current_state.get_reward()

    def backup(self, node: Node, reward: float) -> None:
        """
        回传阶段：将模拟结果沿搜索路径向上回传，更新所有经过节点的统计信息。
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, iterations: int = 1000) -> Node:
        """
        执行指定次数的迭代搜索，返回根节点下访问次数最多的子节点作为最佳选择。
        """
        for _ in tqdm(range(iterations), desc='MCTS Iteration', position=1, leave=False):
            leaf = self.select(self.root)
            reward = self.simulation(leaf.state)
            self.backup(leaf, reward)
        return self.root.best_child(c_param=0)


if __name__ == '__main__':
    test_loader = get_mcts_test_loader()
    first_sample = test_loader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    # 初始化 SAM和RewardModel
    predictor = SamPredictor(sam)
    reward_model = load_model(
        model_name='latest.pth', sample_width=sample_width, sample_height=sample_height)
    utils = Utils(predictor=predictor, reward_model=reward_model)
    results_dir = get_mcts_path()
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'info.log'), 'w') as f:
        f.write(f"Global Info: {utils}\n")
    for data in tqdm(test_loader, desc='Test Image', position=0):
        image = data['image'][0].to(device)
        mask = data['mask'][0].to(device)
        image_id = data['image_id'][0]

        utils.set_image(image)
        initial_state = GameState()
        root = Node(initial_state)

        max_points = utils.max_points
        best_node = root
        for _ in range(max_points):
            mcts = MCTS(best_node.state)
            best_node = mcts.search(iterations=utils.num_simulations)
        points = np.array(best_node.state.all_action_points())
        labels = np.array(best_node.state.all_action_labels())

        reward = best_node.state.get_reward()

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
        # 计算所有f'{image_id}_reward.txt'的均值并追加进去
        reward_results = []
        for file in os.listdir(results_dir):
            if file.endswith('_reward.txt'):
                with open(os.path.join(results_dir, file), 'r') as reward_f:
                    for line in reward_f:
                        if line.startswith("Reward:"):
                            reward = float(line.split(":")[1].strip())
                            reward_results.append(reward)
        mean_reward = np.mean(reward_results)
        f.write(f"Mean Reward: {mean_reward}\n")

    print("Done!")
