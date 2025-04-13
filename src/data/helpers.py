import os
import re


def extract_image_id(image_path):
    '''从文件路径中提取 image_id'''
    # 替换_mask_\d+.png 为 ''
    return re.sub(r'_mask_\d+.png', '', os.path.basename(image_path)).replace('_Segmentation', '').split('.')[0]

def extract_mask_id(mask_file):
    '''从文件路径中提取符合 mask_\d+ 形式的 mask_id'''
    match = re.search(r'mask_\d+', os.path.basename(mask_file))
    if match:
        return match.group(0)
    else:
        raise ValueError(f"无法从路径中提取 mask_id: {mask_file}")

def filter_images(image_files):
    '''过滤掉不符合要求的图片'''
    return image_files
