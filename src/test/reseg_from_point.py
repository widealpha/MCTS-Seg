import os
import numpy as np
from PIL import Image
from segment_anything import SamPredictor
from tqdm import tqdm
from utils.helpers import get_data_path, get_mcts_result_path, get_root_path, load_sam, dataset


def resegment_and_save(image_id, sam_model, input_dir, output_dir, image_dir,  ground_truth_dir, shape=(512, 512)):
    """
    从文件中读取 Best points 和 Best labels，筛选第一个和最后一个点及其对应的 label，
    使用 SAM 重新分割，并保存分割结果，同时计算 IOU 和 Dice。

    :param image_id: 图像 ID
    :param sam_model: SAM 模型实例
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    """
    # 构建输入文件路径
    input_file = os.path.join(
        input_dir, f"{image_id}_best_points_and_reward.txt")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    # 读取文件内容
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 提取 Best points 和 Best labels
    best_points = eval(lines[0].split(":")[1].strip())  # 转换为列表
    best_labels = eval(lines[1].split(":")[1].strip())  # 转换为列表

    # 筛选第一个和最后一个点及其对应的 label
    selected_points = [best_points[0],]
    selected_labels = [best_labels[0],]

    # 加载图像和 ground truth
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, f"{image_id}.png")
    ground_truth_path = os.path.join(
        ground_truth_dir, f"{image_id}.png")
    if not os.path.exists(image_path) or not os.path.exists(ground_truth_path):
        print(f"Image or ground truth for {image_id} does not exist.")
        return

    image = np.array(Image.open(image_path).resize(shape, Image.BICUBIC))
    ground_truth = np.array(Image.open(
        ground_truth_path).resize(shape, Image.NEAREST))

    # 使用 SAM 重新分割
    predictor = SamPredictor(sam_model)
    predictor.set_image(image)
    new_masks, _, _ = predictor.predict(
        np.array(selected_points), np.array(selected_labels), multimask_output=False)
    new_mask = (new_masks[0] * 255).astype(np.uint8)  # 取第一个 mask

    # 保存分割结果
    os.makedirs(output_dir, exist_ok=True)
    result_image_path = os.path.join(
        output_dir, f"{image_id}_result.png")
    mask_path = os.path.join(output_dir, f"{image_id}_mask.png")

    # 保存分割结果图像
    combined_image = np.concatenate(
        (new_mask, ground_truth), axis=1).astype(np.uint8)
    Image.fromarray(combined_image).save(result_image_path)
    Image.fromarray(new_mask).save(mask_path)


def calculate_average_iou_and_dice(input_dir, ground_truth_dir):
    """
    计算平均 IOU 和 Dice 系数
    :param input_dir: 输入文件夹路径
    :param ground_truth_dir: ground truth 文件夹路径
    :return: 平均 IOU 和 Dice 系数
    """
    iou_list = []
    dice_list = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_mask.png"):
            mask_path = os.path.join(input_dir, file_name)
            ground_truth_path = os.path.join(
                ground_truth_dir, file_name.replace("_mask.png", ".png"))
            if not os.path.exists(ground_truth_path):
                continue

            # 加载掩码和 Ground Truth
            mask = np.array(Image.open(mask_path).convert('L'))
            ground_truth = np.array(Image.open(ground_truth_path).convert('L'))

            # 确保掩码和 Ground Truth 的尺寸一致
            if mask.shape != ground_truth.shape:
                ground_truth = np.array(Image.fromarray(ground_truth).resize(mask.shape, Image.NEAREST))

            # 二值化处理（假设阈值为 128）
            mask = (mask > 128).astype(np.uint8)
            ground_truth = (ground_truth > 128).astype(np.uint8)

            # 计算 IOU 和 Dice
            intersection = np.logical_and(mask, ground_truth).sum()
            union = np.logical_or(mask, ground_truth).sum()

            iou = intersection / union if union != 0 else 0
            dice = 2 * intersection / (mask.sum() + ground_truth.sum()) if (mask.sum() + ground_truth.sum()) != 0 else 0

            iou_list.append(iou)
            dice_list.append(dice)

    # 计算平均值
    average_iou = np.mean(iou_list) if iou_list else 0
    average_dice = np.mean(dice_list) if dice_list else 0

    return average_iou, average_dice

def main():
    sam_model = load_sam()  # 加载 SAM 模型
    mcts_path = get_mcts_result_path()
    data_path = get_data_path()
    input_dir = os.path.join(mcts_path)
    output_dir = res = os.path.join(
        get_root_path(), 'result', 'mcts_reseg', dataset)
    ground_truth_dir = os.path.join(data_path, "raw/test/ground_truth")
    image_dir = os.path.join(data_path, "raw/test/image")
    image_files = os.listdir(input_dir)
    image_files.sort()
    image_files = [f for f in image_files if f.endswith("_raw.png")]
    for image_file in tqdm(image_files, desc="Resegmentation"):
        image_id = image_file.replace("_raw.png", '')
        resegment_and_save(image_id, sam_model, input_dir,
                           output_dir, image_dir, ground_truth_dir)
    calculate_average_iou_and_dice(
        os.path.join(output_dir), ground_truth_dir)
    average_iou, average_dice = calculate_average_iou_and_dice(
        os.path.join(output_dir), ground_truth_dir)
    print(f"Average IOU: {average_iou}")
    print(f"Average Dice: {average_dice}")
    with open(os.path.join(output_dir, 'average_iou_and_dice.txt'), 'w') as f:
        f.write(f"Average IOU: {average_iou}\n")
        f.write(f"Average Dice: {average_dice}\n")


if __name__ == '__main__':
    main()
