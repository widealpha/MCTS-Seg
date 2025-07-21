"""
策略模型的数据加载器。
提供图片和随机图片上n个点，以及对应的概率标签。
点的概率：在mask中是1，不在mask中是0。
"""

import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import random

from src.utils.helpers import get_data_path

data_path = get_data_path()


class PolicyDataset(Dataset):
    """
    策略模型的数据集类。
    每个样本包含：
    - 一张图片
    - 图片上随机生成的n个点（坐标）
    - 每个点的概率标签（在mask中为1，不在mask中为0）
    """
    
    def __init__(self, image_dir, min_points=5, max_points=15, image_size=256):
        """
        初始化策略数据集。
        
        Args:
            image_dir: 图片目录路径
            min_points: 每张图片的最小点数
            max_points: 每张图片的最大点数
            image_size: 图片尺寸（假设为正方形）
        """
        self.image_dir = image_dir
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size
        
        # 获取所有图片ID
        file_list = os.listdir(image_dir)
        image_reg = re.compile(r'(.+)_raw\.png$')
        self.image_ids = [f.split('_raw')[0] for f in file_list if re.match(image_reg, f)]
        self.image_ids.sort()
        
        # 为每个图像找到对应的mask文件
        self.image_mask_pairs = []
        for image_id in self.image_ids:
            # 查找该图像的所有mask
            mask_pattern = re.compile(rf'^{image_id}_mask_(\d+)\.png$')
            for f in file_list:
                if mask_pattern.match(f):
                    self.image_mask_pairs.append((image_id, f))
        
        print(f'{image_dir}: {len(self.image_mask_pairs)} 图像-mask对')
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        image_id, mask_filename = self.image_mask_pairs[idx]
        
        # 加载图像和mask
        image_path = os.path.join(self.image_dir, f"{image_id}_raw.png")
        mask_path = os.path.join(self.image_dir, mask_filename)
        
        try:
            # 加载图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transform(image)
            
            # 加载mask
            mask = Image.open(mask_path)
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask = self.transform(mask)
            
            # 生成随机点数量
            n_points = random.randint(self.min_points, self.max_points)
            
            # 生成随机点坐标（归一化到[0,1]范围）
            points = torch.rand(n_points, 2)  # shape: (n_points, 2)
            
            # 计算每个点的概率标签
            labels = self._compute_point_labels(points, mask)
            
            return {
                'image': image,  # shape: (3, image_size, image_size)
                'points': points,  # shape: (n_points, 2)，归一化坐标
                'labels': labels,  # shape: (n_points, 1)，概率标签
                'image_id': image_id,
                'mask_filename': mask_filename
            }
            
        except Exception as e:
            print(f"加载数据时出错 {image_id}: {e}")
            # 返回一个默认样本以避免训练中断
            return self._get_default_sample()
    
    def _compute_point_labels(self, points, mask):
        """
        计算点的概率标签。
        
        Args:
            points: 归一化坐标点 (n_points, 2)
            mask: mask图像 (1, H, W)
        
        Returns:
            labels: 概率标签 (n_points, 1)
        """
        n_points = points.shape[0]
        H, W = mask.shape[1], mask.shape[2]
        
        # 将归一化坐标转换为像素坐标
        pixel_coords = points.clone()
        pixel_coords[:, 0] *= W  # x坐标
        pixel_coords[:, 1] *= H  # y坐标
        pixel_coords = pixel_coords.long()
        
        # 确保坐标在有效范围内
        pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, W-1)
        pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, H-1)
        
        # 提取mask值作为标签
        labels = torch.zeros(n_points, 1)
        for i in range(n_points):
            x, y = pixel_coords[i]
            # mask值 > 0.5 认为是前景（标签为1）
            labels[i, 0] = 1.0 if mask[0, y, x] > 0.5 else 0.0
        
        return labels
    
    def _get_default_sample(self):
        """返回一个默认样本以处理加载错误"""
        n_points = self.min_points
        return {
            'image': torch.zeros(3, self.image_size, self.image_size),
            'points': torch.rand(n_points, 2),
            'labels': torch.zeros(n_points, 1),
            'image_id': 'default',
            'mask_filename': 'default'
        }


def policy_collate_fn(batch):
    """
    自定义的批处理函数，处理不同数量的点。
    
    Args:
        batch: 一个batch的样本列表
    
    Returns:
        批处理后的数据字典
    """
    # 过滤掉None样本
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # 获取batch信息
    batch_size = len(batch)
    max_points = max(item['points'].shape[0] for item in batch)
    
    # 初始化批处理张量
    images = torch.stack([item['image'] for item in batch])
    
    # 创建填充后的点坐标和标签
    padded_points = torch.zeros(batch_size, max_points, 2)
    padded_labels = torch.zeros(batch_size, max_points, 1)
    point_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        n_points = item['points'].shape[0]
        padded_points[i, :n_points] = item['points']
        padded_labels[i, :n_points] = item['labels']
        point_lengths[i] = n_points
    
    # 收集其他信息
    image_ids = [item['image_id'] for item in batch]
    mask_filenames = [item['mask_filename'] for item in batch]
    
    return {
        'images': images,  # (batch_size, 3, H, W)
        'points': padded_points,  # (batch_size, max_points, 2)
        'labels': padded_labels,  # (batch_size, max_points, 1)
        'point_lengths': point_lengths,  # (batch_size,)
        'image_ids': image_ids,
        'mask_filenames': mask_filenames
    }


def get_policy_data_loader(batch_size=16, shuffle=True, test_batch_size=16, 
                          test_shuffle=False, min_points=5, max_points=15, 
                          image_size=256, num_workers=4):
    """
    获取策略模型的数据加载器。
    
    Args:
        batch_size: 训练批次大小
        shuffle: 是否打乱训练数据
        test_batch_size: 测试批次大小
        test_shuffle: 是否打乱测试数据
        min_points: 最小点数
        max_points: 最大点数
        image_size: 图像尺寸
        num_workers: 数据加载工作进程数
    
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """
    # 数据路径
    train_dir = os.path.join(data_path, 'final', 'train')
    test_dir = os.path.join(data_path, 'final', 'test')
    
    # 创建数据集
    train_dataset = PolicyDataset(
        image_dir=train_dir,
        min_points=min_points,
        max_points=max_points,
        image_size=image_size
    )
    
    test_dataset = PolicyDataset(
        image_dir=test_dir,
        min_points=min_points,
        max_points=max_points,
        image_size=image_size
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=policy_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=test_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=policy_collate_fn
    )
    
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # 测试数据加载器
    print("测试策略数据加载器...")
    
    train_loader, test_loader = get_policy_data_loader(
        batch_size=2, 
        min_points=3, 
        max_points=8,
        image_size=256
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        if batch is not None:
            print(f"图像形状: {batch['images'].shape}")
            print(f"点坐标形状: {batch['points'].shape}")
            print(f"标签形状: {batch['labels'].shape}")
            print(f"点数量: {batch['point_lengths']}")
            print(f"图像ID: {batch['image_ids']}")
            
            # 显示第一个样本的统计信息
            first_sample_points = batch['point_lengths'][0].item()
            first_sample_labels = batch['labels'][0, :first_sample_points, 0]
            positive_ratio = (first_sample_labels > 0.5).float().mean()
            print(f"第一个样本正样本比例: {positive_ratio:.3f}")
            break
    
    print("数据加载器测试完成！")
