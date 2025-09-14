import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.utils.helpers import get_data_path

data_path = get_data_path()


class ImageDataset(Dataset):
    def __init__(self, image_dir, per_image_mask=None):
        self.image_dir = image_dir
        file_list = os.listdir(image_dir)
        image_reg = re.compile(r'(.+)_raw\.png$')
        # 提取所有唯一的原始图像ID（排序后）
        self.image_ids = [f.split('_raw')[0]
                          for f in file_list if re.match(image_reg, f)]
        self.image_ids.sort()

        if per_image_mask:
            self.per_image_mask = per_image_mask
        else:
            # 自动计算统一mask数量
            self.per_image_mask = self._validate_and_get_mask_count()
        print(f'{image_dir} {self.per_image_mask} masks per image')

        self.transform = transforms.ToTensor()

    def _validate_and_get_mask_count(self):
        """验证所有image的mask数量一致性并返回统一值"""
        mask_counts = set()
        file_set = set(os.listdir(self.image_dir))

        for image_id in self.image_ids:
            # 统计有效mask数量（需同时存在mask和reward文件）
            valid_masks = 0
            mask_pattern = re.compile(rf'^{image_id}_mask_(\d+)\.png$')
            for f in file_set:
                # 检查mask文件
                if mask_match := mask_pattern.match(f):
                    mask_num = mask_match.group(1)
                    # reward_file = f"{image_id}_mask_{mask_num}_normalized_reward.txt"
                    reward_file = f"{image_id}_mask_{mask_num}_reward.txt"
                    if reward_file in file_set:
                        valid_masks += 1

            mask_counts.add(valid_masks)

        if len(mask_counts) != 1:
            raise ValueError(f"不一致的mask数量, 检测到: {mask_counts}")

        return mask_counts.pop()

    def __len__(self):
        # 每个image_id下的mask对数: C(per_image_mask, 2)
        return len(self.image_ids) * self.per_image_mask * (self.per_image_mask - 1) // 2

    def __getitem__(self, idx):
        # 计算属于哪个image_id
        image_idx = idx // (self.per_image_mask * (self.per_image_mask - 1) // 2)
        pair_idx = idx % (self.per_image_mask * (self.per_image_mask - 1) // 2)
        image_id = self.image_ids[image_idx]
        # 生成所有mask对
        mask_indices = [(i, j) for i in range(self.per_image_mask) for j in range(i+1, self.per_image_mask)]
        i, j = mask_indices[pair_idx]

        image_path = os.path.join(self.image_dir, f"{image_id}_raw.png")
        mask1_path = os.path.join(self.image_dir, f"{image_id}_mask_{i}.png")
        mask2_path = os.path.join(self.image_dir, f"{image_id}_mask_{j}.png")
        reward1_path = os.path.join(self.image_dir, f"{image_id}_mask_{i}_reward.txt")
        reward2_path = os.path.join(self.image_dir, f"{image_id}_mask_{j}_reward.txt")

        try:
            image = self.transform(Image.open(image_path))
            mask1 = self.transform(Image.open(mask1_path))
            mask2 = self.transform(Image.open(mask2_path))
            with open(reward1_path, 'r') as f:
                reward1 = float(f.read().strip())
            with open(reward2_path, 'r') as f:
                reward2 = float(f.read().strip())
        except Exception as e:
            print(f"Error loading pair data for {image_id}: {e}")
            return None

        return {
            'image': image,
            'mask1': mask1,
            'mask2': mask2,
            'reward1': reward1,
            'reward2': reward2,
            'image_id': image_id
        }


def get_data_loader(batch_size=16, shuffle=True, test_batch_size=16, test_shuffle=False, per_image_mask=None):
    # 路径
    train_dir = os.path.join(data_path, 'final', 'train')
    test_dir = os.path.join(data_path, 'final', 'test')

    # 创建数据集实例
    train_dataset = ImageDataset(
        image_dir=train_dir, per_image_mask=per_image_mask)
    # 测试集只使用ground_truth不使用增强数据集
    test_dataset = ImageDataset(
        image_dir=test_dir, per_image_mask=per_image_mask)

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=test_shuffle, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader
