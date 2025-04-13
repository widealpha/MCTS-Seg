import os
from PIL import Image
from tqdm import tqdm

from src.cfg import parse_args


def process_isic2018(input_dir, output_dir):
    """
    处理 ISIC2018 数据集，将图片缩放为指定大小并保存到目标目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    target_size = (parse_args().image_size, parse_args().image_size)
    
    images = [file for file in os.listdir(input_dir) if file.endswith('.jpg') or file.endswith('.png')]
    images.sort()
    for file in tqdm(images, desc=f"Processing ISIC2018"):
        # 打开图片并缩放
        with Image.open(os.path.join(input_dir, file)) as img:
            if "segmentation" in file.lower():
                img_resized = img.resize(target_size, Image.NEAREST)
                file = file.replace('_segmentation.png', '.png')
            else:
                img_resized = img.resize(target_size, Image.BICUBIC)
            output_file_path = os.path.join(output_dir, file.replace('.jpg', '.png'))
            img_resized.save(output_file_path)

if __name__ == '__main__':
    input_base_dir = '/data/share/ISIC2018'
    output_base_dir = 'data/ISIC2018/raw'
    process_isic2018(os.path.join(input_base_dir, 'ISIC2018_Task1-2_Training_Input'),
                     os.path.join(output_base_dir, 'train/image'))
    process_isic2018(os.path.join(input_base_dir, 'ISIC2018_Task1_Training_GroundTruth'),
                     os.path.join(output_base_dir, 'train/gt'))
    process_isic2018(os.path.join(input_base_dir, 'ISIC2018_Task1-2_Test_Input'),
                     os.path.join(output_base_dir, 'test/image'))
    process_isic2018(os.path.join(input_base_dir, 'ISIC2018_Task1_Test_GroundTruth'),
                     os.path.join(output_base_dir, 'test/gt'))
