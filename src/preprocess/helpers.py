import os
import re


def extract_image_id(file_path):
    '''从文件路径中提取 image_id'''
    # 替换_mask_\d+.png 为 ''
    filename = os.path.basename(file_path)
    filename = re.sub(r'_mask_\d+', '', filename)
    
    return filename.split('.')[0]

def extract_mask_id(file_path):
    filename = os.path.basename(file_path)
    mask_id_match = re.search(r'_mask_(\d+)', filename)
    if mask_id_match:
        return mask_id_match.group(1)
    else:
        return '0'

def filter_images(image_files):
    '''过滤掉不符合要求的图片'''
    return image_files
