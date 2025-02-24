import os
import re


def extract_image_id(image_path):
    '''从文件路径中提取 image_id'''
    # 替换_mask_\d+.png 为 ''
    return re.sub(r'_mask_\d+.png', '', os.path.basename(image_path)).replace('_Segmentation', '').split('.')[0]


def filter_images(image_files):
    '''过滤掉不符合要求的图片'''
    return image_files
