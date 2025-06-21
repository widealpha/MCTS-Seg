import os
import numpy as np
from PIL import Image
import torch

from src.models.mcts import get_sam_predictor
from torchvision.io import read_image

from src.utils.helpers import get_result_path


def generate_segmentation(image: torch.Tensor, points: np.ndarray, output_dir: str, image_id: str):
    """
    使用 SAM 生成分割结果并存储到指定路径。

    :param image: 输入图像 (torch.Tensor, shape: C x H x W)
    :param points: 点坐标 (np.ndarray, shape: N x 2)
    :param output_dir: 输出目录
    :param image_id: 图像 ID，用于命名输出文件
    """
    # 获取 SAM 预测器
    predictor = get_sam_predictor()

    # 将图像转换为 numpy 格式并设置到预测器
    image_array = image.permute(1, 2, 0).cpu().numpy()
    predictor.set_image(image_array)

    # 创建标签数组 (全为 1，表示前景点)
    labels = np.ones(points.shape[0], dtype=np.int32)

    # 使用 SAM 预测分割掩码
    with torch.no_grad():
        masks, _, _ = predictor.predict_torch(
            point_coords=torch.from_numpy(
                points).unsqueeze(0).to('cuda'),
            point_labels=torch.from_numpy(
                labels).unsqueeze(0).to('cuda'),
            multimask_output=False,
        )

    # 提取分割掩码并转换为 numpy 格式
    mask = (masks[0][0] > 0.5).cpu().numpy()

    # 保存分割结果
    os.makedirs(output_dir, exist_ok=True)
    raw_image_path = os.path.join(output_dir, f"{image_id}_raw.png")
    mask_path = os.path.join(output_dir, f"{image_id}_mask.png")
    result_path = os.path.join(output_dir, f"{image_id}_result.png")

    # 保存原始图像
    raw_image = (image_array).astype(np.uint8)
    Image.fromarray(raw_image).save(raw_image_path)

    # 保存分割掩码
    mask_image = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_image).save(mask_path)

    # 保存叠加结果
    # combined_image = np.concatenate(
    #     (raw_image, np.expand_dims(mask_image, axis=-1)), axis=1)
    # Image.fromarray(combined_image).save(result_path)

    print(f"Segmentation results saved to {output_dir}")


def main():
    image_path = "/home/kmh/ai/MCTS-Seg/results/ablation/ISIC2016/wo_mcts/ISIC_0000003_raw.png"

    image = Image.open(image_path)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

    # 输入图像路径
    output_dir = os.path.join(get_result_path(), 'segmentation_results')
    image_id = "example_image"
    # 示例点坐标
    points = np.array([[426, 376]])

    # 调用分割函数
    generate_segmentation(image, points, output_dir, image_id)


if __name__ == "__main__":
    main()
