import os
import shutil
import argparse
from data.resize_dataset import resize_and_compare_images
from utils.helpers import get_data_path, get_mcts_path
from data.helpers import extract_image_id


def copy_mask_files(mask_id, overwrite=False):
    source_dir = get_mcts_path()
    target_dir = os.path.join(
        get_data_path(), 'processed', 'train', 'expanded')

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory '{target_dir}'.")
    files = os.listdir(source_dir)
    train_image_ids = {extract_image_id(file) for file in os.listdir(
        os.path.join(get_data_path(), 'raw', 'train', 'image'))}
    files = [file for file in files if file.replace(
        '_mask.png', '') in train_image_ids]
    files.sort()
    for file in files:
        if file.endswith(f"_mask.png"):
            new_file = file.replace("_mask.png", f"_mask_{mask_id}.png")
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, new_file)

            if os.path.exists(target_file) and not overwrite:
                print(f"File '{target_file}' already exists. Skipping.")
            else:
                shutil.copy2(source_file, target_file)
                print(f"Copied '{source_file}' to '{target_file}'.")


def recalculate_mask_rewards():
    data_path = get_data_path()
    raw_path = os.path.join(data_path, 'raw')
    raw_image_dir = os.path.join(raw_path, 'train', 'image')
    processed_path = os.path.join(data_path, 'processed')
    ground_truth_dir = os.path.join(raw_path, 'train', 'ground_truth')
    # 整合ground_truth以及上述三/四种mask的目录
    expanded_dir = os.path.join(processed_path, 'train', 'expanded')
    # 对上述数据应用新的reward算法并缩放的保存结果的目录
    resized_dir = os.path.join(processed_path, 'train', 'resized')
    image_size = (512, 512)
    # image_size = None

    resize_and_compare_images(
        in_dir=expanded_dir, out_dir=resized_dir, raw_dir=raw_image_dir, size=image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy mask files to train directory.")
    parser.add_argument("mask_id", type=int, help="ID of the mask to copy.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files.")
    args = parser.parse_args()

    copy_mask_files(args.mask_id, args.overwrite)
    # copy_mask_files(5, False)
    recalculate_mask_rewards()
