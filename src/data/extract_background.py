import os
import traceback
from PIL import Image

import numpy as np
from tqdm import tqdm

from data.helpers import extract_image_id, filter_images


def extract_bg(in_dir, out_dir, ground_truth_dir):
    image_folder = in_dir
    output_folder = out_dir
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(
        image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = filter_images(image_files)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        image_id = extract_image_id(image_file)
        try:
            with Image.open(image_path).convert('L') as img:
                img_array = np.array(img, dtype=np.uint8)
                # 获取img_array中==0的部分，并设置为255保存到mask_image
                mask_image = Image.fromarray(
                    np.where(img_array == 0, 255, 0).astype(np.uint8))
                mask_filename = f"{image_id}_mask_0.png"
                mask_image.save(os.path.join(
                    output_folder, mask_filename))
                # 匹配以image_id开头的ground_truth文件
                ground_truth_files = [f for f in os.listdir(
                    ground_truth_dir) if f.startswith(image_id)]
                if len(ground_truth_files) == 0:
                    print(f"Ground truth for {image_file} does not exist.")
                    continue
                ground_truth_file = ground_truth_files[0]
                ground_truth_path = os.path.join(ground_truth_dir, ground_truth_file)

                if not os.path.exists(ground_truth_path):
                    print(f"Ground truth for {image_file} does not exist.")
                    continue
                
                ground_truth = Image.open(ground_truth_path).convert('L')
                ground_truth = np.array(ground_truth)
                # 计算iou
                calculate_iou = np.sum(
                    np.logical_and(ground_truth, img_array)) / np.sum(np.logical_or(ground_truth, img_array))
                if calculate_iou != 0:
                    if (calculate_iou >= 0.05):
                        print(f"Image {image_file} has non-zero IoU: {calculate_iou}")
                # 背景的reward理应为0
                with open(os.path.join(output_folder, f"{image_id}_best_score.txt"), 'w') as f:
                    f.write(f"0\n")
        except Exception as e:
            print(f"\nError processing image {image_file}: {e}\n")
            traceback.print_exc()
