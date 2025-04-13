import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.utils.helpers import get_data_path

data_path = get_data_path()


class MCTSDataset(Dataset):
    def __init__(self, image_dir):
        """
        :param image_dir: 原图像所在目录
        """
        self.image_dir = image_dir
        file_list = os.listdir(image_dir)
        image_reg = re.compile(r'(.+)_raw\.png$')
        # 提取所有唯一的原始图像ID（排序后）
        self.image_ids = [f.split('_raw')[0]
                          for f in file_list if re.match(image_reg, f)]
        self.image_ids.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = f'{self.image_ids[idx]}'
        image_path = os.path.join(self.image_dir, f"{image_id}_raw.png")
        mask_path = os.path.join(self.image_dir, f"{image_id}_mask_0.png")

        try:
            image = self.transform(Image.open(image_path))
            mask = self.transform(Image.open(mask_path))
        except Exception as e:
            print(f"Error loading data for {image_id}: {e}")
            return None

        return {
            'image': image,
            'mask': mask,
            'image_id': image_id
        }


def get_mcts_test_loader(batch_size=1, shuffle=False):
    data_dir = os.path.join(data_path, 'final', 'test')
    dataset = MCTSDataset(image_dir=data_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader
