import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.cfg import parse_args
from src.utils.helpers import get_data_path

data_path = get_data_path()


class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        file_list = os.listdir(image_dir)
        # 提取所有唯一的原始图像ID（排序后）
        self.image_ids = [f.replace('.png', '').replace(
            '.jpg', '') for f in file_list]
        self.image_ids.sort()

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        try:
            image_size = parse_args().image_size
            image = self.transform(Image.open(image_path).resize((image_size, image_size)))
            mask = self.transform(Image.open(mask_path).resize((image_size, image_size)))
        except Exception as e:
            print(f"Error loading data for {image_id}: {e}")
            return None
        return {
            'image_id': image_id,
            'image': image,
            'mask': mask,
        }


def get_baseline_dataloader(batch_size=16, shuffle=True, test_batch_size=16):
    # 路径
    train_dir = os.path.join(data_path, 'raw', 'train')
    test_dir = os.path.join(data_path, 'raw', 'test')

    # 创建数据集实例
    train_dataset = ImageDataset(image_dir=os.path.join(
        train_dir, 'image'), mask_dir=os.path.join(train_dir, 'gt'))
    # 测试集只使用ground_truth不使用增强数据集
    test_dataset = ImageDataset(image_dir=os.path.join(
        test_dir, 'image'), mask_dir=os.path.join(test_dir, 'gt'))

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader
