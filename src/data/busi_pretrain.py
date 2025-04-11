import os
import random
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings

from utils.helpers import setup_seed

def normalize_image(slice_data, method='minmax'):

    if method == 'minmax':
        # 将数据标准化到[0, 1]之间
        # slice_data = (slice_data - np.min(slice_data)) / \
        #     (np.max(slice_data) - np.min(slice_data))
        MIN_BOUND = 0
        MAX_BOUND = np.max(slice_data)
        slice_data[slice_data > MAX_BOUND] = MAX_BOUND
        slice_data[slice_data < MIN_BOUND] = MIN_BOUND
        slice_data = (slice_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    elif method == 'zscore':
        # 使用均值和标准差进行标准化
        slice_data = (slice_data - np.mean(slice_data)) / np.std(slice_data)
    return slice_data


def crop_nonzero_region(img):
    """
    裁剪图片中的非全零区域
    参数:
      img: numpy 数组，支持2D或3D数据
    返回:
      裁剪后的图片
    """
    # 使用 np.nonzero() 获取所有非零元素的索引
    nonzero_indices = np.nonzero(img)

    # 如果图像全为零，则直接返回原图
    if len(nonzero_indices[0]) == 0:
        return img

    # 对于每个维度，找到最小和最大的索引
    min_indices = [np.min(idx) for idx in nonzero_indices]
    max_indices = [np.max(idx) for idx in nonzero_indices]

    # 根据数据维度进行裁剪
    if img.ndim == 2:
        cropped = img[min_indices[0]:max_indices[0] +
                      1, min_indices[1]:max_indices[1]+1]
    elif img.ndim == 3:
        cropped = img[min_indices[0]:max_indices[0]+1,
                      min_indices[1]:max_indices[1]+1,
                      min_indices[2]:max_indices[2]+1]
    else:
        raise ValueError("不支持的数据维度: {}".format(img.ndim))

    return cropped


def process(mode='train', type='benign', shape=(1024, 1024)):
    # Define the paths
    input_dir = f'/home/kmh/Downloads/Dataset_BUSI_with_GT/{type}'
    output_image_dir = f'data/BUSI-{type}/raw/{mode}/image'
    output_gt_dir = f'data/BUSI-{type}/raw/{mode}/ground_truth'
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)
    images = os.listdir(input_dir)
    images = [file.replace('_mask.png', '.png')
              for file in images if file.endswith('_mask.png')]
    setup_seed()
    random.shuffle(images)
    # 训练集测试集1:2划分
    test_images = images[::3]
    train_images = [d for i, d in enumerate(images) if i % 3 != 0]
    if mode == 'train':
        images = train_images
    else:
        images = test_images
    images = sorted(images)
    for image in tqdm(images, desc='Processing image'):
        image_name = image
        gt_name = image.replace('.png', '_mask.png')
        with Image.open(os.path.join(input_dir, image_name)) as img, \
                Image.open(os.path.join(input_dir, gt_name)) as gt:
            img = img.resize(shape, Image.BICUBIC)
            gt = gt.resize(shape, Image.NEAREST)
            output_name = image_name.replace(' (', '').replace(')', '')
            gt_output_path = os.path.join(
                output_gt_dir, output_name)
            image_output_path = os.path.join(
                output_image_dir, output_name)
            gt.save(gt_output_path)
            img.save(image_output_path)


if __name__ == '__main__':
    process(mode='train', type='benign')
    process(mode='test', type='benign')
    process(mode='train', type='malignant')
    process(mode='test', type='malignant')
