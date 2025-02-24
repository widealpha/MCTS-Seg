import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils.helpers import get_data_path

data_path = get_data_path()


class ISICDataset(Dataset):
    def __init__(self, image_dir, per_image_mask=4):
        """
        :param image_dir: 原图像所在目录
        """
        self.image_dir = image_dir
        file_list = os.listdir(image_dir)
        self.image_ids = [file.split('_raw')[0] for file in file_list if re.match(
            r'^ISIC_\d+_raw\.jpg$', file)]
        self.image_ids.sort()
        self.per_image_mask = per_image_mask
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_ids) * self.per_image_mask

    def __getitem__(self, idx):
        image_id = f'{self.image_ids[idx // self.per_image_mask]}'
        mask_id = f'{image_id}_mask_{idx % self.per_image_mask}'
        image_path = os.path.join(self.image_dir, f"{image_id}_raw.jpg")
        mask_path = os.path.join(self.image_dir, f"{mask_id}.png")
        reward_path = os.path.join(
            self.image_dir, f"{mask_id}_normalized_reward.txt")

        try:
            image = self.transform(Image.open(image_path))
            mask = self.transform(Image.open(mask_path))
            with open(reward_path, 'r') as f:
                reward = float(f.read().strip())
        except Exception as e:
            print(f"Error loading data for {image_id}: {e}")
            return None

        return {
            'image': image,
            'mask': mask,
            'reward': reward,
            'image_id': image_id
        }


def get_data_loader(batch_size=2, shuffle=True, test_batch_size=4, test_shuffle=False):
    # 路径
    train_dir = os.path.join(data_path, 'processed/train/resized')
    test_dir = os.path.join(data_path, 'processed/test/resized')

    # 创建数据集实例
    train_dataset = ISICDataset(image_dir=train_dir)
    test_dataset = ISICDataset(image_dir=test_dir)

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=test_shuffle)
    return train_dataloader, test_dataloader
