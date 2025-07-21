from __future__ import annotations
from datetime import datetime
import math
import os
import random
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
from segment_anything import SamPredictor
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from src.cfg import parse_args
from src.models.model import RewardPredictionModel
from src.data.mcts_loader import get_mcts_test_loader
from src.models.msa_predictor import MSAPredictor
from src.utils.helpers import calculate_dice, calculate_iou, get_checkpoints_path, get_mcts_result_path, load_sam_adapter, setup_seed, load_sam, get_dataset, get_device
import csv

setup_seed()

checkpoints_path = get_checkpoints_path()
utils: Utils = None
device = get_device()
dataset = get_dataset()


def get_sam_predictor():
    """
    获取 SAM 预测器
    """
    args = parse_args()
    if args.bone == 'sam':
        sam = load_sam()
        sam_predictor = SamPredictor(sam)
    elif args.bone == 'msa':
        sam = load_sam_adapter()
        sam_predictor = MSAPredictor(sam)
    return sam_predictor


class RewardCache:
    def __init__(self):
        self.cache = {}

    def _key(self, actions: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """
        将 Set[List[int]] 转换为可哈希的 Tuple[Tuple[int, ...], ...]，并排序以保证一致性。
        """
        return tuple(sorted(tuple(action) for action in actions))

    def get_reward_cache(self, actions: List[List[int]]) -> Optional[float]:
        """
        获取缓存中的奖励值，如果不存在则返回 None。
        """
        return self.cache.get(self._key(actions))

    def set_reward_cache(self, actions: List[List[int]], reward: float):
        """
        设置缓存中的奖励值。
        """

        self.cache[self._key(actions)] = reward

    def clear_cache(self):
        """
        清空缓存。
        """
        self.cache.clear()


class Utils:
    def __init__(self, predictor: SamPredictor, reward_model: RewardPredictionModel):
        # 初始化 SAM 预测器和 RewardModel
        self.predictor = predictor
        self.reward_model = reward_model
        # 设置 MCTS 参数
        # 预测max_points个点
        self.max_points = 2
        # 每次网格划分为K*K块
        self.grid_size = 8
        # 网格划分的深度信息
        self.max_depth = None
        # 每次模拟的次数
        self.num_simulations = 8 * self.grid_size * self.grid_size
        # 模拟时候选定的随机点数量
        self.simulation_actions = 1
        # 允许使用背景点
        self.enable_background = False
        self.use_ground_truth = False
        self.use_random_ground_truth = False
        self.float_reward = True
        self.points = []
        self.labels = []
        self.cache = RewardCache()
        self.baseline_reward = None

    def set_image(self, image: torch.Tensor, image_id: str = None, ground_truth=None):
        self.image = image  # tensor(C, H, W)
        # 将(C,H,W)转换为(H,W,C)
        self.image_array = image.permute(1, 2, 0).cpu().numpy()
        # 扩充为(1, C, H, W)
        self.single_image = image.unsqueeze(0)
        # 扩充为(B, C, H, W)
        self.batch_size = 4
        # 扩充为(B, H, W, C)
        self.batch_image = image.unsqueeze(
            0).repeat(self.batch_size, 1, 1, 1).to(device)
        # 获取图像的宽和高
        self.width = image.shape[2]
        self.height = image.shape[1]
        self.predictor.set_image(self.image_array)
        # 网格划分的最大深度
        self.grid = DynamicGrid(
            n=min(self.width, self.height), m=self.grid_size)
        self.max_depth = math.ceil(
            math.log(min(self.width, self.height), self.grid_size))
        # raw = math.log(min(self.width, self.height), self.grid_size)
        # self.max_depth = math.ceil(raw) - (1 if raw == int(raw) else 2)

        self.image_id = image_id
        if self.use_ground_truth and ground_truth is not None:
            gt_array = ground_truth[0].cpu().numpy()
            if self.use_random_ground_truth:
                point_number = 1
                points = []
                labels = []
                # 获取gt_array中的point_number / 2个=0的点
                # 获取gt_array中的point_number - (point_number / 2)个>0的点
                y_indices_0, x_indices_0 = np.where(gt_array == 0)
                y_indices_1, x_indices_1 = np.where(gt_array > 0)
                if len(y_indices_0) > 0:
                    indices_0 = np.random.choice(len(y_indices_0), min(
                        point_number // 2, len(y_indices_0)), replace=False)
                    for idx in indices_0:
                        points.append([x_indices_0[idx], y_indices_0[idx]])
                        labels.append(0)

                if len(y_indices_1) > 0:
                    indices_1 = np.random.choice(len(y_indices_1), min(
                        point_number - (point_number // 2), len(y_indices_1)), replace=False)
                    for idx in indices_1:
                        points.append([x_indices_1[idx], y_indices_1[idx]])
                        labels.append(1)
                self.points = points
                self.labels = labels
            else:
                # 计算ground_truth的box中心
                y_indices, x_indices = np.where(gt_array > 0)
                if len(y_indices) == 0:
                    print(f"No foreground pixels found in ground truth for image.")
                # 使用np.where(gt_array > 0)对应的矩形的中心点
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                # 使用重心作为前景点
                self.points = [[center_x, center_y]]
                self.labels = [1]

    def hyper_param_str(self):
        """
        返回当前 MCTS 的超参数设置字符串。
        """
        return (
            f"m{self.max_points}"
            f"g{self.grid_size}"
            f"s{self.num_simulations}"
            f"a{self.simulation_actions}"
            f"bg{1 if self.enable_background else 0}"
            f"t{1 if self.use_ground_truth else 0}"
            f"rt{1 if self.use_random_ground_truth else 0}"
            f"f{1 if self.float_reward else 0}"
        )

    def __str__(self):
        return (
            f"""GlobalInfo(
    max_points           = {self.max_points},
    grid_size            = {self.grid_size},
    num_simulations      = {self.num_simulations},
    simulation_actions   = {self.simulation_actions},
    enable_background    = {self.enable_background},
    use_ground_truth     = {self.use_ground_truth},
    random_ground_truth  = {self.use_random_ground_truth}
    float_reward         = {self.float_reward},
)""")


def load_model(model_name, sample_width, sample_height):
    model_path = os.path.join(checkpoints_path, model_name)
    model = RewardPredictionModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict(predictor, reward_model, points: np.array, labels: np.array, image: torch.tensor):
    with torch.no_grad():
        torch_points = None
        torch_labels = None
        if points is not None and labels is not None:
            # 将 points 和 labels 转换为 torch tensor
            torch_points = torch.from_numpy(points).unsqueeze(0).to(device)
            torch_labels = torch.from_numpy(labels).unsqueeze(0).to(device)
        new_masks, *_ = predictor.predict_torch(
            torch_points,
            torch_labels,
            multimask_output=False
        )

        mask = (new_masks > 0.5).float()
        rewards = reward_model(image, mask)
        # 使用 RewardModel 计算 reward
        return mask.to(device), rewards.to(device)  # (B, 1, H, W)


def sam_seg_cal_reward(predictor, reward_model, points, labels, image, ground_truth, image_id, output_dir):
    """
    使用 SAM 进行分割，结合 ground truth 计算 reward，reward 直接使用 IOU，结果保存在 results/mcts 文件夹下。
    :param predictor: SAM 预测器
    :param points: 分割点
    :param labels: 分割标签
    :param image: 输入图像 tensor(C, H, W)
    :param ground_truth: ground truth 掩码 tensor(1, H, W)
    :param image_id: 图像 ID
    """
    # 使用 SAM 生成新的 mask
    with torch.no_grad():
        masks, rewards = predict(predictor=predictor,
                                 reward_model=reward_model,
                                 points=points, labels=labels,
                                 image=image.unsqueeze(0))
        reward = rewards.item()
        new_mask = masks[0]
    image_array = image.cpu().numpy()
    mask_array = new_mask[0].cpu().numpy()
    gt_array = ground_truth[0].cpu().numpy()
    # 计算 IOU 和Dice
    iou = calculate_iou(mask_array, gt_array)
    dice = calculate_dice(mask_array, gt_array)

    # 保存结果
    results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, f'{image_id}_raw.png')
    result_path = os.path.join(results_dir, f'{image_id}_result.png')
    mask_path = os.path.join(results_dir, f'{image_id}_mask.png')
    # iou_path = os.path.join(results_dir, f'{image_id}_iou.txt')
    # reward_path = os.path.join(results_dir, f'{image_id}_reward.txt')

    # 保存分割结果和 mask
    raw_image = (image_array * 255).astype(np.uint8).transpose(1, 2, 0)
    new_mask_image = (mask_array * 255).astype(np.uint8)
    ground_truth_image = (gt_array * 255).astype(np.uint8)
    combined_image = np.concatenate(
        (new_mask_image, ground_truth_image), axis=1)
    Image.fromarray(raw_image).save(image_path)
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
    # # 保存 IOU
    # with open(iou_path, 'w') as f:
    #     f.write(f'{iou}')
    # with open(reward_path, 'w') as f:
    #     f.write(f'{reward}')

    return iou, dice, reward


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

        # 确保返回的 size 不小于 1
        size = max(size, 1)
        return int(round(top_left_x)), int(round(top_left_y)), size

    def get_coordinate(self, selection: List[int]) -> Tuple[int, int]:
        """
        根据选择列表返回当前区域中心点的坐标（整数）。
        如果网格大小不足以支持划分，则直接返回左上角坐标。
        """
        top_left_x, top_left_y, size = self._get_region(selection)
        if size <= 1:
            # 如果网格大小不足以划分，返回左上角坐标
            return int(round(top_left_x)), int(round(top_left_y))
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

        如果当前区域大小不足以支持 m*m 的划分，则返回实际可划分的数量。
        """
        actions: Dict[str, List[Tuple[List[int], Tuple[int, int]]]] = {
            "siblings": [], "children": []}

        # 当前区域
        cur_top_left_x, cur_top_left_y, cur_size = self._get_region(selection)

        # 如果当前区域大小为 1，则无法进一步细分
        if cur_size == 1:
            return actions

        # 动态调整划分数量，确保不会超过当前网格的大小
        actual_m = min(self.m, int(cur_size))  # 实际划分数量不能超过当前网格大小

        # 子动作：对当前区域进一步划分
        for i in range(1, actual_m * actual_m + 1):
            new_selection = selection + [i]
            index = i - 1
            row = index // actual_m
            col = index % actual_m
            cell_size = cur_size / actual_m
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
            for i in range(1, actual_m * actual_m + 1):
                if i == current_choice:
                    continue  # 跳过当前选择
                new_selection = selection[:-1] + [i]
                index = i - 1
                row = index // actual_m
                col = index % actual_m
                cell_size = parent_size / actual_m
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
        判定标准：若当前区域的网格大小刚好为 1，则认为无法再划分，即为终局。
        """
        _, _, size = self._get_region(selection)
        return size == 1  # 修改为判断网格大小是否刚好为 1

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
        action_history_label: List[int] = None,
        current_action: List[int] = None,
        current_action_label: int = 1,
    ) -> None:
        self.action_history: List[List[int]
                                  ] = action_history if action_history is not None else []
        self.action_history_label: List[int] = action_history_label if action_history_label is not None else [
        ]
        self.current_action: List[int] = current_action if current_action is not None else [
        ]
        self.current_action_label = current_action_label
        self.reward: float = None

    def is_terminal(self) -> bool:
        """
        判断当前状态是否为终局。
        终局的判定依据：使用当前的 current_action 对应的区域，
        若该区域继续划分后每个子块尺寸小于 1，则认为无法再细分，即为终局。
        """
        return utils.grid.is_fully_subdivided(self.current_action)

    def get_possible_moves(self) -> List[Tuple[List[int], int]]:
        """
        返回当前状态下所有合法的下一步动作（新的选择路径）。
        包括：
          - 当前区域进一步细分后的各子块（children）
          - 若当前状态非根状态，还包括同级替换（siblings）的选择
        """
        actions = utils.grid.get_possible_actions(self.current_action)
        moves: List[Tuple[List[int], int]] = []
        # 是否允许背景点选择
        if utils.enable_background:
            if self.current_action:
                moves.extend([(move, 1)
                             for move, coord in actions["children"]])
                moves.extend([(move, 0)
                             for move, coord in actions["children"]])
                moves.extend([(move, 1)
                             for move, coord in actions["siblings"]])
                moves.extend([(move, 0)
                             for move, coord in actions["siblings"]])
            else:
                moves.extend([(move, 1)
                             for move, coord in actions["children"]])
                moves.extend([(move, 0)
                             for move, coord in actions["children"]])

        else:
            moves.extend([(move, 1) for move, coord in actions["children"]])
            if self.current_action:
                moves.extend([(move, 1)
                             for move, coord in actions["siblings"]])
        return moves

    def get_possible_children_moves(self) -> Tuple[List[int], int]:
        """
        返回当前状态下所有合法的下一步动作（新的选择路径）。
        包括：
          - 当前区域进一步细分后的各子块（children）
        """
        actions = utils.grid.get_possible_actions(self.current_action)
        moves: List[List[int]] = [(move, self.current_action_label)
                                  for move, coord in actions["children"]]
        return moves

    def apply_moves(self, moves: List[Tuple[List[int], int]]) -> GameState:
        """
        根据给定的动作（新的选择路径）生成新的状态。
        更新方式：将当前状态的 current_action 追加到 action_history 中，
        并将新的 move 作为新的 current_action 返回。
        """
        new_history = self.action_history.copy()
        new_history_label = self.action_history_label.copy()
        if self.current_action:
            new_history.append(self.current_action)
            new_history_label.append(self.current_action_label)
        for move in moves[:-2]:
            new_history.append(move[0])
            new_history_label.append(move[1])
        return GameState(action_history=new_history, action_history_label=new_history_label, current_action=moves[-1][0], current_action_label=moves[-1][1])

    def apply_move(self, move: Tuple[List[int], int]) -> GameState:
        """
        根据给定的动作（新的选择路径）生成新的状态。
        更新方式：将当前状态的 current_action 追加到 action_history 中，
        并将新的 move 作为新的 current_action 返回。
        """
        return self.apply_moves([move])
        # new_history = self.action_history.copy()
        # new_history_label = self.action_history_label.copy()
        # if self.current_action:  # 记录历史（根状态 current_action 为空时不记录）
        # new_history.append(self.current_action)
        # new_history_label.append(self.current_action_label)
        # return GameState(action_history=new_history, action_history_label=new_history_label, current_action=move[0], current_action_label=move[1])

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
        """
        all_action_labels = self.action_history_label.copy()
        if self.current_action:
            all_action_labels.append(self.current_action_label)
        return all_action_labels
        # all_actions = self.action_history.copy()
        # if self.current_action:
        #     all_actions.append(self.current_action)
        # return [1 for _ in all_actions]

    def get_reward(self) -> float:
        """
        计算 reward：
          1. 根据 all_action_points() 与 all_action_labels() 获取所有点与标签，
          2. 使用 SAM 预测器生成新的 mask，
          3. 利用 RewardModel 根据 single_image 和生成的 mask 计算 reward，
          4. 将计算结果保存到 self.reward 并返回。
        """
        cache_key = [*self.action_history, self.current_action]
        reward = utils.cache.get_reward_cache(cache_key)
        baseline_reward = utils.baseline_reward
        if baseline_reward is None:
            _, rewards = predict(predictor=utils.predictor,
                                 reward_model=utils.reward_model,
                                 points=None,
                                 labels=None,
                                 image=utils.single_image)
            baseline_reward = rewards.item()
            utils.baseline_reward = baseline_reward

        if reward is not None:
            return reward

        points = np.array(utils.points + self.all_action_points())
        labels = np.array(utils.labels + self.all_action_labels())
        # 使用 SAM 生成新的 mask
        with torch.no_grad():
            _, rewards = predict(predictor=utils.predictor,
                                 reward_model=utils.reward_model,
                                 points=points,
                                 labels=labels,
                                 image=utils.single_image)
            # 使用 RewardModel 计算 reward
            reward = rewards.item()
            utils.cache.set_reward_cache(cache_key, reward)
            # 返回 1 或 -1，表示是否超过基线奖励
            return reward if utils.float_reward else (1 if reward > baseline_reward else -1)


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
        self.untried_moves: List[Tuple[List[int], int]
                                 ] = state.get_possible_moves()

    def expand(self) -> Node:
        """
        从未尝试的动作中选择一个扩展子节点
        """
        move = self.untried_moves.pop(
            np.random.randint(len(self.untried_moves)))
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

    def __init__(self, root: Node) -> None:
        self.root = root
        # else:
        #     self.root: Node = Node(state=root_state)

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
        # current_state = state
        # while not current_state.is_terminal():
        #     possible_moves = current_state.get_possible_children_moves()
        #     if not possible_moves:
        #         break
        #     move = random.sample(possible_moves, max(
        #                          min(utils.simulation_actions, len(possible_moves) // 4), 1))
        #     current_state = current_state.apply_moves(move)
        # return current_state.get_reward()
        rewards = []
        for i in range(utils.simulation_actions):
            current_state = state
            while not current_state.is_terminal():
                possible_moves = current_state.get_possible_children_moves()
                if not possible_moves:
                    break
                move = [random.choice(possible_moves)]
                current_state = current_state.apply_moves(move)
            rewards.append(current_state.get_reward())
        # 计算平均奖励
        if len(rewards) > 0:
            reward = sum(rewards) / len(rewards)
        else:
            reward = 0.0
        return reward if utils.float_reward else int(reward > 0.85)

    def backup(self, node: Node, reward: float) -> None:
        """
        回传阶段：将模拟结果沿搜索路径向上回传，更新所有经过节点的统计信息。
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, iterations: int = 1000, depth: int = 0) -> Node:
        """
        执行指定次数的迭代搜索，返回根节点下访问次数最多的子节点作为最佳选择。
        """
        for _ in tqdm(range(iterations), desc=f'{utils.image_id} MCTS Iteration Depth {depth}', position=1, leave=False):
            leaf = self.select(self.root)
            reward = self.simulation(leaf.state)
            self.backup(leaf, reward)
        return self.root.best_child(c_param=0)


def add_action_to_node(root: Node,  action_history: List[List[int]] = None,
                       action_history_label: List[int] = None):
    """从root节点向下，修改所有的state将，action_history的内容和label都添加到state最前面"""
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        # Prepend the action_history and labels to the current state
        current_node.state.action_history = action_history + \
            current_node.state.action_history
        current_node.state.action_history_label = action_history_label + \
            current_node.state.action_history_label
        current_node.state.reward = None
        # Add children to the queue for further processing
        queue.extend(current_node.children)


def run_mcts(results_dir):
    print(f"Start MCTS Test Dataset:{dataset} ...")
    test_loader = get_mcts_test_loader()
    first_sample = test_loader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    # 初始化 SAM和RewardModel
    predictor = get_sam_predictor()
    reward_model = load_model(
        model_name='latest.pth', sample_width=sample_width, sample_height=sample_height)
    global utils
    utils = Utils(predictor=predictor, reward_model=reward_model)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'info.log'), 'w') as f:
        f.write(f"Global Info: {utils}\n")
    for data in tqdm(test_loader, desc='Test Image', position=0):
        image = data['image'][0].to(device)
        mask = data['mask'][0].to(device)
        image_id = data['image_id'][0]

        utils.set_image(image=image, image_id=image_id, ground_truth=mask)
        points_path = []
        labels_path = []
        for i in range(utils.max_points):
            initial_state = GameState()
            root = Node(initial_state)
            best_node = root
            for j in range(utils.max_depth):
                mcts = MCTS(best_node)
                best_node = mcts.search(
                    iterations=utils.num_simulations, depth=j + 1)
            # 只加入最底层的点，忽略上层点
            all_action_points = best_node.state.all_action_points()
            all_action_labels = best_node.state.all_action_labels()
            # 记录路径
            points_path.append(all_action_points)
            labels_path.append(all_action_labels)
            # 取最后一个点
            utils.points.append(all_action_points[-1])
            utils.labels.append(all_action_labels[-1])

        points = np.array(utils.points)
        labels = np.array(utils.labels)

        reward = best_node.state.get_reward()

        iou, dice, reward = sam_seg_cal_reward(
            predictor=predictor, reward_model=reward_model,
            points=points, labels=labels,
            image=image, ground_truth=mask, image_id=image_id,
            output_dir=results_dir
        )
        # 将最佳点和奖励写入文件
        result_file_path = os.path.join(
            results_dir, f'{image_id}_result.txt')
        with open(result_file_path, 'w') as f:
            f.write(f"Points: {utils.points}, Path: {points_path}\n")
            f.write(f"Labels: {utils.labels}, Path: {labels_path}\n")
            f.write(f"IoU: {iou}\n")
            f.write(f"Dice: {dice}\n")
            f.write(f"Reward: {reward}\n")
        # 后处理，清除全局数据
        utils.points.clear()
        utils.labels.clear()
        utils.cache.clear_cache()
        points_path.clear()
        labels_path.clear()


def calculate_iou_dice(results_dir):
    # 初始化结果列表
    iou_results = []
    dice_results = []
    reward_results = []
    result_files = [f for f in os.listdir(
        results_dir) if f.endswith('_result.txt')]

    # 遍历结果目录，读取 IoU 和 Reward 文件
    for file in result_files:
        with open(os.path.join(results_dir, file), 'r') as result_f:
            for line in result_f:
                if line.startswith("Reward:"):
                    reward = float(line.split(":")[1].strip())
                    reward_results.append(reward)
                elif line.startswith("IoU:"):
                    iou = float(line.split(":")[1].strip())
                    iou_results.append(iou)
                elif line.startswith("Dice:"):
                    dice = float(line.split(":")[1].strip())
                    dice_results.append(dice)

    # 计算统计信息
    mean_iou = np.mean(iou_results)
    var_iou = np.var(iou_results)
    std_iou = np.std(iou_results)

    mean_dice = np.mean(dice_results)
    var_dice = np.var(dice_results)
    std_dice = np.std(dice_results)

    mean_reward = np.mean(reward_results)
    var_reward = np.var(reward_results)
    std_reward = np.std(reward_results)

    # 将统计信息写入 CSV 文件
    csv_file_path = os.path.join(results_dir, 'results_summary.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头
        writer.writerow(['Metric', 'Mean', 'Variance', 'Standard Deviation'])
        # 写入 IoU、Dice 和 Reward 的统计信息
        writer.writerow(['IoU', mean_iou, var_iou, std_iou])
        writer.writerow(['Dice', mean_dice, var_dice, std_dice])
        writer.writerow(['Reward', mean_reward, var_reward, std_reward])

    # 打印结果
    print(f"Results saved to {csv_file_path}")
    print(f"Mean IoU: {mean_iou}, Variance IoU: {var_iou}, Std IoU: {std_iou}")
    print(
        f"Mean Dice: {mean_dice}, Variance Dice: {var_dice}, Std Dice: {std_dice}")
    print(
        f"Mean Reward: {mean_reward}, Variance Reward: {var_reward}, Std Reward: {std_reward}")


def get_result_path():
    """
    获取 MCTS 结果保存路径。
    根据当前时间戳创建一个新的目录，格式为 'results/mcts/年-月-日_时-分-秒'。
    """
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args = parse_args()
    result_dir = os.path.join(get_mcts_result_path(), Utils(
        None, None).hyper_param_str(), f"{args.bone}-{current_time}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def main():
    result_path = get_result_path()
    run_mcts(results_dir=result_path)
    calculate_iou_dice(results_dir=result_path)
    print("Done!")


if __name__ == '__main__':
    main()
