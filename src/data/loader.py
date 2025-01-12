import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.utils.helpers import get_root_path

root_path = get_root_path()


class ISBIDataset(Dataset):
    def __init__(self, image_dir):
        """
        :param image_dir: 原图像所在目录
        """
        self.image_dir = image_dir
        file_list = os.listdir(image_dir)
        self.image_files = [file.split('.')[0] for file in file_list if re.match(
            r'^ISIC_\d{7}\.png$', file)]
        self.per_image_mask = 4
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files) * self.per_image_mask

    def __getitem__(self, idx):
        image_id = f'{self.image_files[idx // self.per_image_mask]}'
        mask_id = f'{image_id}_mask_{idx % self.per_image_mask}'
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        mask_path = os.path.join(self.image_dir, f"{mask_id}.png")
        image_feature_path = os.path.join(self.image_dir, f"{image_id}.pt")
        mask_feature_path = os.path.join(self.image_dir, f"{mask_id}.pt")
        iou_path = os.path.join(self.image_dir, f"{mask_id}_iou.txt")
        # ground_truth_path = os.path.join(self.image_dir, f"{image_id}_mask_3_512.png")
        try:
            image = self.transform(Image.open(image_path))
            # image = self.resize(image)
            mask = self.transform(Image.open(mask_path).convert('RGB'))
            # mask = self.resize(mask)
            # image_feature = torch.load(image_feature_path)[0]
            # mask_feature = torch.load(mask_feature_path)[0]
            image_feature = torch.tensor([])
            mask_feature = torch.tensor([])

            # ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            with open(iou_path) as f:
                iou = float(f.readline())
                iou = torch.tensor(iou, dtype=torch.float32)

            return image_feature, mask_feature, iou, image_id, mask_id, image, mask
        except IOError as e:
            print(e)
            print(image_id)
            print(mask_id)
            raise e

    # def resize(self, image):
    #     return cv2.resize(image, (512, 512))

def get_isbi_dataloader(train=True):
    data_dir = os.path.join(root_path, '../../data/processed/train')
    dataset = ISBIDataset(image_dir=data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    return dataloader
    

def data_loader():
    # 路径
    train_dir = "../data/processed/ISBI2016_ISIC/train"
    test_dir = "../data/processed/ISBI2016_ISIC/test"
    # ground_truth_dir = "../data/raw/ISBI2016_ISIC/train/ground_truth"

    # 创建数据集实例
    train_dataset = ISBIDataset(image_dir=train_dir)
    test_dataset = ISBIDataset(image_dir=test_dir)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_dataloader, test_dataloader
