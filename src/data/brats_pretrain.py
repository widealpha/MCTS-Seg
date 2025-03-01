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
        slice_data = (slice_data - np.min(slice_data)) / \
            (np.max(slice_data) - np.min(slice_data))
    elif method == 'zscore':
        # 使用均值和标准差进行标准化
        slice_data = (slice_data - np.mean(slice_data)) / np.std(slice_data)
    return slice_data


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
    # 分离测试集和训练集,90%训练集，10%测试集,按照文件夹名字排序，每9个训练一个测试
    test_subdirs = subdirs[::10]
    train_subdirs = [d for i, d in enumerate(subdirs) if i % 10 != 0]
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
                    # 增强去
                    seg_slice[seg_slice == 4] = 1
                    # 水肿
                    seg_slice[seg_slice == 2] = 0
                    if (seg_slice.max() == 0):
                        continue
                    seg_slice = (seg_slice - np.min(seg_slice)) / \
                        (np.max(seg_slice) - np.min(seg_slice))
                    seg_slice = (seg_slice * 255).astype(np.uint8)
                    image_slice = image_data[:, :, i]
                    # # 获取image_slice除去0意外的最小值
                    # min_value = np.min(image_slice[image_slice > 0])
                    # max_value = np.max(image_slice)
                    # # 仅仅改变非0的值映射到1-255之间
                    # image_slice[image_slice > 0] = (image_slice[image_slice > 0] - 1) / \
                    #     (max_value - min_value) * 254 + 1
                    # 使用minmax标准化，并映射到0-255之间
                    image_slice = normalize_image(image_slice, 'minmax')
                    image_slice = (image_slice * 255).astype(np.uint8)
                    seg_img = Image.fromarray(seg_slice, mode='L')
                    image_img = Image.fromarray(image_slice, mode='L')
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
