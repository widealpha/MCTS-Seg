# 生成SAM对图像的原始分割文件
import json
import os

import numpy as np
import torch
from utils.helpers import load_sam, get_root_path, setup_seed
from tqdm import tqdm
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
setup_seed()
sam = load_sam()

root_path = get_root_path()


def sam_auto_mask():
    mask_generator = SamAutomaticMaskGenerator(sam)
    # 定义图片文件夹路径
    image_folder = os.path.join(
        root_path, 'data/raw/train/ISBI2016_ISIC/image')
    output_folder = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/auto_masks')
    json_output_file = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/auto_masks/mask_metadata.json')
    metadata = []  # 用于存储所有 mask 信息
    os.makedirs(output_folder, exist_ok=True)
    # 检查路径是否存在
    if os.path.exists(image_folder):
        # 获取所有图片文件列表
        image_files = [f for f in os.listdir(
            image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        # image_files = ['ISIC_0009860.jpg']
        # 使用 tqdm 遍历所有图片文件
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            # 打开并处理图片
            try:
                with Image.open(image_path) as img:
                    # 转换为模型所需的格式 (如果需要)
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.uint8)
                    # 生成 mask
                    masks = mask_generator.generate(img_array)
                    # 保存所有 mask 并编号
                    for i, mask_data in enumerate(masks):
                        mask = mask_data['segmentation']
                        bbox = mask_data['bbox']
                        area = mask_data['area']
                        predicted_iou = mask_data['predicted_iou']
                        point_coords = mask_data['point_coords']
                        stability_score = mask_data['stability_score']
                        crop_box = mask_data['crop_box']

                        # 保存 mask 图像
                        mask_image = Image.fromarray(
                            mask.astype('uint8') * 255)
                        mask_filename = f"{os.path.splitext(image_file)[0]}_mask_{i}.png"
                        mask_image.save(os.path.join(
                            output_folder, mask_filename))

                        # 添加元数据
                        metadata.append({
                            'image_file': image_file,
                            'mask_file': mask_filename,
                            'bbox': bbox,
                            'area': area,
                            'predicted_iou': predicted_iou,
                            'point_coords': point_coords,
                            'stability_score': stability_score,
                            'crop_box': crop_box
                        })
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                # 保存所有的元数据到 JSON 文件
        with open(json_output_file, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
    else:
        print(f"The folder '{image_folder}' does not exist.")


def sam_point_mask(point_number, grid_size):
    '''
    将图像网格划分为grid_size * grid_size的网格，
    依据ground_truth从每个网格中随机选取point_number个在其中的点，生成点标注的mask
    '''
    image_folder = os.path.join(
        root_path, 'data/raw/train/ISBI2016_ISIC/image')
    ground_truth_folder = os.path.join(
        root_path, 'data/raw/train/ISBI2016_ISIC/ground_truth')
    output_folder = os.path.join(
        root_path, 'data/processed/train/ISBI2016_ISIC/point_masks')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(image_folder) or not os.path.exists(ground_truth_folder):
        print(
            f"One of the folders '{image_folder}' or '{ground_truth_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    predictor = SamPredictor(sam)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        ground_truth_path = os.path.join(
            ground_truth_folder, image_file.replace('.jpg', '_Segmentation.png'))

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

                for i, mask in enumerate(masks):
                    mask_image = Image.fromarray(mask.astype('uint8') * 255)
                    mask_filename = f"{os.path.splitext(image_file)[0]}_point_mask_{i}.png"
                    mask_image.save(os.path.join(output_folder, mask_filename))

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")


if __name__ == '__main__':
    sam_point_mask(1, 20)
