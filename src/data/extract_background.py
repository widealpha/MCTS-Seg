import os
import traceback
from PIL import Image

import numpy as np
from tqdm import tqdm

from data.helpers import extract_image_id, filter_images


def extract_bg(in_dir, out_dir):
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
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                # 获取img_array中==0的部分，并设置为255保存到mask_image
                mask_image = Image.fromarray(
                    np.where(img_array == 0, 255, 0).astype(np.uint8))
                mask_filename = f"{image_id}_mask_0.png"
                mask_image.save(os.path.join(
                    output_folder, mask_filename))

                with open(os.path.join(output_folder, f"{image_id}_best_score.txt"), 'w') as f:
                    f.write(f"0\n")
        except Exception as e:
            print(f"\nError processing image {image_file}: {e}\n")
            traceback.print_exc()
