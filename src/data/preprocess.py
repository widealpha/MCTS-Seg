import os
from utils.helpers import get_root_path
from data.sam_seg import sam_auto_mask, sam_point_mask, sam_point_mask_all_points
from data.select_image_best_rewards import select_best_rewards_all_image, select_best_rewards_image
from data.extract_connected_part import extract_largest_connected_component
from data.expand_dataset import copy_best_rewards
from data.resize_dataset import resize_and_compare_images

root_path = get_root_path()


def generate_data(train=False, use_best_point=False):
    # 定义目录路径
    if train:
        data_type = 'train'
    else:
        data_type = 'test'

    expanded_dir = os.path.join(
        root_path, f'data/processed/{data_type}/expanded')
    raw_image_dir = os.path.join(
        root_path, f'data/raw/{data_type}/ISBI2016_ISIC/image')
    ground_truth_dir = os.path.join(
        root_path, f'data/raw/{data_type}/ISBI2016_ISIC/ground_truth')
    auto_masks_dir = os.path.join(
        root_path, f'data/processed/{data_type}/ISBI2016_ISIC/auto_masks')
    all_point_masks_dir = os.path.join(
        root_path, f'data/processed/{data_type}/ISBI2016_ISIC/all_point_masks')
    point_masks_dir = os.path.join(
        root_path, f'data/processed/{data_type}/ISBI2016_ISIC/point_masks')
    connected_point_masks_dir = os.path.join(
        root_path, f'data/processed/{data_type}/ISBI2016_ISIC/connected_point_masks')
    resized_dir = os.path.join(
        root_path, f'data/processed/{data_type}/resized')

    # 生成 SAM 自动掩码
    sam_auto_mask(in_dir=raw_image_dir, out_dir=auto_masks_dir)

    # 选择最佳奖励图像
    select_best_rewards_image(in_dir=auto_masks_dir, ground_truth_dir=ground_truth_dir,
                              out_dir=os.path.join(auto_masks_dir, 'best_rewards'))
    if use_best_point:
        # 生成最优点掩码
        sam_point_mask_all_points(
            grid_size=20,
            in_dir=raw_image_dir, out_dir=all_point_masks_dir, ground_truth_dir=ground_truth_dir)
        # 选择最佳奖励图像
        select_best_rewards_all_image(in_dir=all_point_masks_dir, ground_truth_dir=ground_truth_dir,
                                      out_dir=os.path.join(all_point_masks_dir, 'best_rewards'))
        # 提取最大连通部分
        extract_largest_connected_component(
            in_dir=all_point_masks_dir, out_dir=connected_point_masks_dir)
        # 选择最佳奖励图像
        select_best_rewards_image(in_dir=connected_point_masks_dir, ground_truth_dir=ground_truth_dir,
                                  out_dir=os.path.join(connected_point_masks_dir, 'best_rewards'))
    else:
        # 生成随机单点掩码
        sam_point_mask(point_number=1, grid_size=20, in_dir=raw_image_dir, out_dir=point_masks_dir,
                       ground_truth_dir=ground_truth_dir)
        # 选择最佳奖励图像
        select_best_rewards_image(in_dir=point_masks_dir, ground_truth_dir=ground_truth_dir,
                                  out_dir=os.path.join(point_masks_dir, 'best_rewards'))
        # 提取最大连通部分
        extract_largest_connected_component(
            in_dir=point_masks_dir, out_dir=connected_point_masks_dir)
        # 选择最佳奖励图像
        select_best_rewards_image(in_dir=connected_point_masks_dir, ground_truth_dir=ground_truth_dir,
                                  out_dir=os.path.join(connected_point_masks_dir, 'best_rewards'))

    # 复制最佳奖励文件
    copy_best_rewards(in_dir=ground_truth_dir, out_dir=expanded_dir,
                      ground_truth_dir=ground_truth_dir, index=0)
    copy_best_rewards(in_dir=os.path.join(auto_masks_dir, 'best_rewards'), out_dir=expanded_dir,
                      ground_truth_dir=ground_truth_dir, index=1)
    copy_best_rewards(in_dir=os.path.join(connected_point_masks_dir, 'best_rewards'), out_dir=expanded_dir,
                      ground_truth_dir=ground_truth_dir, index=2)
    copy_best_rewards(in_dir=os.path.join(point_masks_dir, 'best_rewards'), out_dir=expanded_dir,
                      ground_truth_dir=ground_truth_dir, index=3)

    # 调整图像大小并生成奖励
    resize_and_compare_images(
        in_dir=expanded_dir, out_dir=resized_dir, raw_dir=raw_image_dir, size=(1024, 1024))


if __name__ == '__main__':
    # generate_data(train=True)  # 生成训练数据
    generate_data(train=False, use_best_point=True)  # 生成测试数据
