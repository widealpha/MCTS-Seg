# 生成SAM对图像的原始分割文件
import json
import os
import traceback

import numpy as np
from tqdm import tqdm
from PIL import Image
from segment_anything import SamPredictor
from src.preprocess.helpers import filter_images, extract_image_id
from src.utils.helpers import calculate_dice, load_sam

def get_sam_predictor():
    sam = load_sam()
    return SamPredictor(sam)


def rewards_function(mask, ground_truth):
    """
    计算 mask 和 ground_truth 之间的 IoU（Intersection over Union）。
    :param mask: 预测的 mask
    :param ground_truth: ground truth mask
    :return: IoU 值
    """
    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)
    return calculate_dice(mask, ground_truth)


def sam_random_point_mask(fg_point_num, bg_point_num, in_dir, ground_truth_dir, out_dir):
    '''
    生成随机点标注的mask
    '''
    image_folder = in_dir
    ground_truth_folder = ground_truth_dir
    output_folder = os.path.join(out_dir)
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    predictor = get_sam_predictor()

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

        try:
            with Image.open(image_path) as img, Image.open(ground_truth_path) as gt:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                gt_array = np.array(gt.convert('L'), dtype=np.uint8)

                # height, width = gt_array.shape

                points = []
                labels = []
                # 获取gt_array中的point_number / 2个=0的点
                # 获取gt_array中的point_number - (point_number / 2)个>0的点
                y_indices_0, x_indices_0 = np.where(gt_array == 0)
                y_indices_1, x_indices_1 = np.where(gt_array > 0)
                if len(y_indices_0) > 0:
                    indices_0 = np.random.choice(len(y_indices_0), min(
                        bg_point_num, len(y_indices_0)), replace=False)
                    for idx in indices_0:
                        points.append([x_indices_0[idx], y_indices_0[idx]])
                        labels.append(0)

                if len(y_indices_1) > 0:
                    indices_1 = np.random.choice(len(y_indices_1), min(
                        fg_point_num, len(y_indices_1)), replace=False)
                    for idx in indices_1:
                        points.append([x_indices_1[idx], y_indices_1[idx]])
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
                    reward = rewards_function(mask, gt_array)
                    if reward > best_reward:
                        best_mask = mask
                        best_reward = reward
                        best_reward_index = i
                # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png
                mask_image = Image.fromarray(best_mask.astype('uint8') * 255)
                mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                mask_image.save(os.path.join(
                    output_folder, mask_filename))

                with open(os.path.join(output_folder, f"{image_id}_score.txt"), 'w') as f:
                    f.write(f"{best_reward}\n")
        except Exception as e:
            print(f"\nError processing image {image_file}: {e}\n")
            print(points)
            traceback.print_exc()


def sam_baseline_point_mask(in_dir, ground_truth_dir, out_dir):
    '''
    使用ground_truth的重心作为前景点生成mask
    '''
    image_folder = in_dir
    ground_truth_folder = ground_truth_dir
    output_folder = out_dir
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    predictor = get_sam_predictor()

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

        try:
            with Image.open(image_path) as img, Image.open(ground_truth_path) as gt:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                gt_array = np.array(gt.convert('L'), dtype=np.uint8)

                # 计算ground_truth的重心
                y_indices, x_indices = np.where(gt_array > 0)
                if len(y_indices) == 0:
                    print(
                        f"No foreground pixels found in ground truth for {image_file}.")
                    continue
                # center_y = int(np.mean(y_indices))
                # center_x = int(np.mean(x_indices))

                # 使用np.where(gt_array > 0)对应的矩形的中心点
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                # 使用重心作为前景点
                points = np.array([[center_x, center_y]])
                labels = np.array([1])

                predictor.set_image(img_array)
                masks, _, _ = predictor.predict(points, labels)

                # 遍历所有masks取一个最佳的mask
                best_mask = None
                best_reward = 0
                best_reward_index = 0
                for i, mask in enumerate(masks):
                    reward = rewards_function(mask, gt_array)
                    if reward > best_reward:
                        best_mask = mask
                        best_reward = reward
                        best_reward_index = i

                # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png
                mask_image = Image.fromarray(best_mask.astype('uint8') * 255)
                mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                mask_image.save(os.path.join(output_folder, mask_filename))

                with open(os.path.join(output_folder, f"{image_id}_score.txt"), 'w') as f:
                    f.write(f"{best_reward}\n")
        except Exception as e:
            print(f"\nError processing image {image_file}: {e}\n")
            traceback.print_exc()


def sam_baseline_auto_mask(in_dir, ground_truth_dir, out_dir):
    '''
    不使用任何点输入，模型自己分割
    '''
    image_folder = in_dir
    ground_truth_folder = ground_truth_dir
    output_folder = out_dir
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    predictor = get_sam_predictor()

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

        try:
            with Image.open(image_path) as img, Image.open(ground_truth_path) as gt:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                gt_array = np.array(gt.convert('L'), dtype=np.uint8)

                predictor.set_image(img_array)
                masks, _, _ = predictor.predict()

                # 遍历所有masks取一个最佳的mask
                best_mask = None
                best_reward = 0
                best_reward_index = 0
                for i, mask in enumerate(masks):
                    reward = rewards_function(mask, gt_array)
                    if reward > best_reward:
                        best_mask = mask
                        best_reward = reward
                        best_reward_index = i

                # 保存best_mask到指定的文件，文件名规则为原来的文件名_mask_index.png
                mask_image = Image.fromarray(best_mask.astype('uint8') * 255)
                mask_filename = f"{image_id}_mask_{best_reward_index}.png"
                mask_image.save(os.path.join(output_folder, mask_filename))

                with open(os.path.join(output_folder, f"{image_id}_score.txt"), 'w') as f:
                    f.write(f"{best_reward}\n")
        except Exception as e:
            print(f"\nError processing image {image_file}: {e}\n")
            traceback.print_exc()


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

    predictor = get_sam_predictor()

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

                with open(os.path.join(output_folder, f"{image_id}_score.txt"), 'w') as f:
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

    predictor = get_sam_predictor()

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

                    with open(os.path.join(output_folder, f"{image_id}_score.txt"), 'w') as f:
                        f.write(f"{best_reward}\n")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
