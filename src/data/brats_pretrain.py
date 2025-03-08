import os
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from matplotlib import pyplot as plt

# 选择标准化方法：


def normalize_image(slice_data, method='minmax'):

    if method == 'minmax':
        # 将数据标准化到[0, 1]之间
        # slice_data = (slice_data - np.min(slice_data)) / \
        #     (np.max(slice_data) - np.min(slice_data))
        MIN_BOUND = 0
        MAX_BOUND = 700
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


def process(mode='train'):
    # Define the paths
    input_dir = '/home/kmh/brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    output_image_dir = f'data/brats2020/raw/{mode}/image'
    output_gt_dir = f'data/brats2020/raw/{mode}/ground_truth'
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)
    subdirs = os.listdir(input_dir)
    subdirs = sorted(subdirs)
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join(input_dir, d))]
    # 训练集测试集1:2划分
    test_subdirs = subdirs[::3]
    train_subdirs = [d for i, d in enumerate(subdirs) if i % 3 != 0]
    if mode == 'train':
        subdirs = train_subdirs
    else:
        subdirs = test_subdirs
    subdirs = sorted(subdirs)
    for subdir in tqdm(subdirs, desc='Processing subdirectories'):
        for file in os.listdir(os.path.join(input_dir, subdir)):
            if '_seg.nii' in file.lower():
                seg_file_path = os.path.join(input_dir, subdir, file)
                image_file_path = seg_file_path.replace('_seg', '_t1ce')
                seg_data = nib.load(seg_file_path).get_fdata()
                image_data = nib.load(image_file_path).get_fdata()
                image_id = os.path.splitext(file)[0].replace('_seg', '')
                # step设置为4

                for i in range(0, seg_data.shape[2], 4):
                    
                    seg_slice = seg_data[:, :, i]
                    seg_slice = seg_slice.astype(np.uint8)
                    # 核心区
                    seg_slice[seg_slice == 1] = 1
                    # 没有3出现
                    seg_slice[seg_slice == 3] = 1
                    # 增强区
                    seg_slice[seg_slice == 4] = 1
                    # 水肿
                    seg_slice[seg_slice == 2] = 0
                    if np.sum(seg_slice == 1) < 4:
                        continue
                    if seg_slice.max() == 0:
                        continue
                    seg_slice = (seg_slice * 255).astype(np.uint8)
                    image_slice = image_data[:, :, i]
                    image_shape = image_slice.shape

                    nonzero_indices = np.nonzero(image_slice)
                    # 对于每个维度，找到最小和最大的索引
                    min_indices = [np.min(idx) for idx in nonzero_indices]
                    max_indices = [np.max(idx) for idx in nonzero_indices]

                    image_slice = image_slice[min_indices[0]:max_indices[0] +
                                              1, min_indices[1]:max_indices[1]+1]
                    seg_slice = seg_slice[min_indices[0]:max_indices[0] +
                                          1, min_indices[1]:max_indices[1]+1]
                    # # 获取image_slice除去0意外的最小值
                    # min_value = np.min(image_slice[image_slice > 0])
                    # max_value = np.max(image_slice)
                    # # 仅仅改变非0的值映射到1-255之间
                    # image_slice[image_slice > 0] = (image_slice[image_slice > 0] - 1) / \
                    #     (max_value - min_value) * 254 + 1
                    # 使用minmax标准化，并映射到0-255之间
                    image_slice = normalize_image(image_slice, 'minmax')
                    image_slice = (image_slice * 255).astype(np.uint8)
                    seg_img = Image.fromarray(seg_slice, mode='L').resize(image_shape, Image.NEAREST)
                    image_img = Image.fromarray(image_slice, mode='L').resize(image_shape, Image.BICUBIC)
                    output_filename = f"{image_id}_{i}.png"
                    seg_output_path = os.path.join(
                        output_gt_dir, output_filename)
                    image_output_path = os.path.join(
                        output_image_dir, output_filename)
                    seg_img.save(seg_output_path)
                    image_img.save(image_output_path)


if __name__ == '__main__':
    process('train')
    process('test')
