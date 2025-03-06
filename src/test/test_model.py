import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from src.data.loader import get_data_loader
from src.data.mcts_loader import get_mcts_test_loader
from src.models.mcts import load_model, calculate_iou
from src.utils.helpers import device, get_mcts_path, get_test_model_path, load_sam, setup_seed
from src.models.model import RewardPredictionModel
from tqdm import tqdm
from segment_anything import SamPredictor
from PIL import Image, ImageDraw

setup_seed()
sam = load_sam()

def draw_image(image, points, labels, output_path, image_id):
    """
    绘制图像和分割点
    :param image: 图像 (H, W)
    :param points: 分割点 (N, 2)
    :param labels: 分割标签 (N)
    """
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    radius = 6
    for point, label in zip(points, labels):
        x, y = point
        color = 'green' if label == 0 else 'red'
        draw.ellipse((x - radius,
                        y - radius, x + radius, y + radius), outline=color, width=2)
    image.save(output_path)


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
    results_dir = get_test_model_path()
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


def main():
    test_loader = get_mcts_test_loader()
    first_sample = test_loader.dataset[0]
    sample_shape = first_sample['image'].shape
    sample_width, sample_height = sample_shape[1], sample_shape[2]
    # 初始化 SAM和RewardModel
    predictor = SamPredictor(sam)
    reward_model = load_model(
        model_name='latest.pth', sample_width=sample_width, sample_height=sample_height).to(device)
    for batch in tqdm(test_loader, desc=f'Test'):
        image = batch['image'][0].to(device)
        mask = batch['mask'][0].to(device)
        image_id = batch['image_id'][0]

        # 获取图像和掩码的尺寸
        _, _, height, width = image.shape
        grid_size = 10  # 可以根据需要调整网格大小
        grid_height = height // grid_size
        grid_width = width // grid_size

        # 随机选择一个网格的中心点
        grid_x = np.random.randint(0, grid_size)
        grid_y = np.random.randint(0, grid_size)
        center_x = grid_x * grid_width + grid_width // 2
        center_y = grid_y * grid_height + grid_height // 2

        # 获取中心点的掩码值
        center_value = mask[0, center_y, center_x].item()
        label = 1 if center_value > 0 else 0
        sam_seg_cal_reward(predictor,
                           np.array([(center_x, center_y)]),
                           np.array([label]),
                           mask[0].cpu().numpy(),
                           image_id)
        mask = torch.Tensor(new_masks[0]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reward = reward_model(batch['image'], mask)
        score = reward.item()


if __name__ == '__main__':
    main()
