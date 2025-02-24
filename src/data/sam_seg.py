# 生成SAM对图像的原始分割文件
import json
import os

import numpy as np
from utils.helpers import load_sam, get_data_path, setup_seed
from tqdm import tqdm
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from helpers import filter_images, extract_image_id

setup_seed()
sam = load_sam()


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 Dice 系数。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: Dice 系数
    """
    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)
    intersection = np.logical_and(mask, ground_truth)
    dice_score = (2 * np.sum(intersection)) / \
        (np.sum(mask) + np.sum(ground_truth) + 1e-7)
    return dice_score


def sam_auto_mask(in_dir, out_dir, ground_truth_dir):
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=32)
    # 定义图片文件夹路径
    image_folder = in_dir
    output_folder = os.path.join(out_dir, 'best_rewards')
    ground_truth_folder = ground_truth_dir
    json_output_file = os.path.join(out_dir, 'mask_metadata.json')
    metadata = []  # 用于存储所有 mask 信息
    os.makedirs(output_folder, exist_ok=True)
    # 检查路径是否存在
    if os.path.exists(image_folder):
        # 获取所有图片文件列表
        image_files = [f for f in os.listdir(
            image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        image_files = filter_images(image_files)
        # 使用 tqdm 遍历所有图片文件
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            image_id = extract_image_id(image_file)

            # 打开并处理图片
            try:
                with Image.open(image_path) as img:
                    # 转换为模型所需的格式 (如果需要)
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.uint8)
                    # 生成 mask
                    masks = mask_generator.generate(img_array)
                    ground_truth_files = [f for f in os.listdir(
                        ground_truth_dir) if f.startswith(image_id)]
                    if len(ground_truth_files) == 0:
                        print(f"Ground truth for {image_file} does not exist.")
                        continue
                    ground_truth_file = ground_truth_files[0]
                    ground_truth_path = os.path.join(
                        ground_truth_folder, ground_truth_file)
                    ground_truth = Image.open(ground_truth_path).convert('L')
                    ground_truth = np.array(ground_truth, dtype=np.uint8)
                    # 遍历所有masks取一个最佳的mask
                    best_mask = None
                    best_reward = 0
                    best_reward_index = 0
                    for i, mask_data in enumerate(masks):
                        mask = mask_data['segmentation']
                        reward = rewards_function(mask, ground_truth)
                        if reward > best_reward:
                            best_mask = mask
                            best_reward = reward
                            best_reward_index = i
                    # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png
                    mask_image = Image.fromarray(
                        best_mask.astype('uint8') * 255)
                    mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                    mask_image.save(os.path.join(output_folder, mask_filename))

                    with open(os.path.join(output_folder, f"{image_id}_best_score.txt"), 'w') as f:
                        f.write(f"{best_reward}\n")

                    # 保存所有 mask 并编号
                    # for i, mask_data in enumerate(masks):
                    #     mask = mask_data['segmentation']
                    #     bbox = mask_data['bbox']
                    #     area = mask_data['area']
                    #     predicted_iou = mask_data['predicted_iou']
                    #     point_coords = mask_data['point_coords']
                    #     stability_score = mask_data['stability_score']
                    #     crop_box = mask_data['crop_box']

                    #     # 保存 mask 图像
                    #     mask_image = Image.fromarray(
                    #         mask.astype('uint8') * 255)
                    #     mask_filename = f"{os.path.splitext(image_file)[0]}_mask_{i}.png"

                    #     mask_image.save(os.path.join(
                    #         output_folder, mask_filename))

                    #     # 添加元数据
                    #     metadata.append({
                    #         'image_file': image_file,
                    #         'mask_file': mask_filename,
                    #         'bbox': bbox,
                    #         'area': area,
                    #         'predicted_iou': predicted_iou,
                    #         'point_coords': point_coords,
                    #         'stability_score': stability_score,
                    #         'crop_box': crop_box
                    #     })
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                # 保存所有的元数据到 JSON 文件
        with open(json_output_file, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
    else:
        print(f"The folder '{image_folder}' does not exist.")


def sam_point_mask(point_number, grid_size, in_dir, ground_truth_dir, out_dir):
    '''
    将图像网格划分为grid_size * grid_size的网格，
    依据ground_truth从每个网格中随机选取point_number个在其中的点，生成点标注的mask
    '''
    image_folder = in_dir
    ground_truth_folder = ground_truth_dir
    output_folder = os.path.join(out_dir, 'best_rewards')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    predictor = SamPredictor(sam)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        image_id = extract_image_id(image_file)
        
        ground_truth_files = [f for f in os.listdir(
            ground_truth_folder) if f.startswith(image_id)]
        if len(ground_truth_files) == 0:
            print(f"Ground truth for {image_id} does not exist.")
            continue
        ground_truth_file = ground_truth_files[0]
        ground_truth_path = os.path.join(
            ground_truth_folder, ground_truth_file)

        try:
            with Image.open(image_path) as img, Image.open(ground_truth_path) as gt:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                gt_array = np.array(gt.convert('L'), dtype=np.uint8)

                height, width = gt_array.shape
                grid_height = height // grid_size
                grid_width = width // grid_size

                points = []
                labels = []

                for i in range(grid_size):
                    for j in range(grid_size):
                        grid = gt_array[i * grid_height:(
                            i + 1) * grid_height, j * grid_width:(j + 1) * grid_width]
                        y_indices, x_indices = np.where(grid > 0)
                        if len(y_indices) > 0:
                            indices = np.random.choice(len(y_indices), min(
                                point_number, len(y_indices)), replace=False)
                            for idx in indices:
                                points.append(
                                    [j * grid_width + x_indices[idx], i * grid_height + y_indices[idx]])
                                labels.append(1)

                points = np.array(points)
                labels = np.array(labels)

                predictor.set_image(img_array)
                masks, _, _ = predictor.predict(points, labels)
                # 遍历所有masks取一个最佳的mask
                best_mask = None
                best_reward = 0
                best_reward_index = 0
                for i, mask in enumerate(masks):
                    # mask = mask_data['segmentation']
                    reward = rewards_function(mask, gt_array)
                    if reward > best_reward:
                        best_mask = mask
                        best_reward = reward
                        best_reward_index = i
                # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png

                mask_image = Image.fromarray(
                    best_mask.astype('uint8') * 255)
                mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                mask_image.save(os.path.join(
                    output_folder, mask_filename))

                with open(os.path.join(output_folder, f"{image_id}_best_score.txt"), 'w') as f:
                    f.write(f"{best_reward}\n")

                # for i, mask in enumerate(masks):
                #     mask_image = Image.fromarray(mask.astype('uint8') * 255)
                #     mask_filename = f"{os.path.splitext(image_file)[0]}_point_mask_{i}.png"
                #     mask_image.save(os.path.join(output_folder, mask_filename))

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")


def sam_point_mask_all_points(grid_size, in_dir, ground_truth_dir, out_dir):
    '''
    将图像网格划分为grid_size * grid_size的网格，
    依据ground_truth从每个网格中随机选取point_number个在其中的点，生成点标注的mask
    '''
    image_folder = in_dir
    ground_truth_folder = ground_truth_dir
    output_folder = os.path.join(out_dir, 'best_rewards')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    predictor = SamPredictor(sam)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        image_id = extract_image_id(image_file)
        ground_truth_files = [f for f in os.listdir(
            ground_truth_folder) if f.startswith(image_id)]
        if len(ground_truth_files) == 0:
            print(f"Ground truth for {image_file} does not exist.")
            continue
        ground_truth_file = ground_truth_files[0]
        ground_truth_path = os.path.join(
            ground_truth_folder, ground_truth_file)
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {image_file} does not exist.")
            continue

        try:
            with Image.open(image_path) as img, Image.open(ground_truth_path) as gt:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                gt_array = np.array(gt.convert('L'), dtype=np.uint8)

                height, width = gt_array.shape
                grid_height = height // grid_size
                grid_width = width // grid_size

                points = []
                labels = []

                for i in range(grid_size):
                    for j in range(grid_size):
                        grid = gt_array[i * grid_height:(
                            i + 1) * grid_height, j * grid_width:(j + 1) * grid_width]
                        center_y = i * grid_height + grid_height // 2
                        center_x = j * grid_width + grid_width // 2
                        points.append([center_x, center_y])
                        labels.append(
                            1 if gt_array[center_y, center_x] > 0 else 0)

                points = np.array(points)
                labels = np.array(labels)

                predictor.set_image(img_array)

                image_output_folder = os.path.join(
                    output_folder, os.path.splitext(image_file)[0])
                os.makedirs(image_output_folder, exist_ok=True)

                for idx, point in enumerate(points):
                    masks, _, _ = predictor.predict(
                        np.array([point]), np.array([labels[idx]]))
                    # 遍历所有masks取一个最佳的mask
                    best_mask = None
                    best_reward = 0
                    best_reward_index = 0
                    for i, mask_data in enumerate(masks):
                        mask = mask_data['segmentation']
                        reward = rewards_function(mask, gt_array)
                        if reward > best_reward:
                            best_mask = mask
                            best_reward = reward
                            best_reward_index = i
                    # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png
                    mask_image = Image.fromarray(
                        best_mask.astype('uint8') * 255)
                    mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                    mask_image.save(os.path.join(
                        output_folder, mask_filename))

                    with open(os.path.join(output_folder, f"{image_id}_best_score.txt"), 'w') as f:
                        f.write(f"{best_reward}\n")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")


if __name__ == '__main__':
    data_path = get_data_path()
    in_dir = os.path.join(data_path, 'raw', 'train', 'image')
    out_dir = os.path.join(data_path, 'processed', 'train', 'auto_masks')
    ground_truth_dir = os.path.join(data_path, 'raw', 'train', 'ground_truth')

    sam_auto_mask(in_dir=in_dir,
                  ground_truth_dir=ground_truth_dir,
                  out_dir=out_dir)
