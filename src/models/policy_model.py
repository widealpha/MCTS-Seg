"""
基于UNet架构的MCTS分割策略模型。
接受一张图片和一个n×2维向量作为输入，输出n×1的概率值用于排序。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

try:
    # 尝试相对导入（当作为模块导入时）
    from .unet_parts import DoubleConv, Down, Up, OutConv
except ImportError:
    # 如果相对导入失败，尝试绝对导入（当直接运行时）
    import sys
    import os
    
    # 添加src目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    from models.unet_parts import DoubleConv, Down, Up, OutConv


class MixedDimensionDataset:
    """
    支持混合n维度的数据集类（向量维度固定为2）。
    用于训练能处理不同动作数量的自适应模型。
    """
    
    def __init__(self, images, vectors_list, labels_list, max_actions=None):
        """
        参数:
            images: 图像列表或张量
            vectors_list: 向量列表，每个元素形状为 (n_i, 2)
            labels_list: 标签列表，每个元素形状为 (n_i, 1)
            max_actions: 最大动作数量（用于填充）
        """
        self.images = images
        self.vectors_list = vectors_list
        self.labels_list = labels_list
        
        # 计算最大动作数量
        if max_actions is None:
            self.max_actions = max(v.shape[0] for v in vectors_list)
        else:
            self.max_actions = max_actions
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        vector = self.vectors_list[idx]
        label = self.labels_list[idx]
        
        n_actions = vector.shape[0]
        
        # 创建填充的向量和标签（向量维度固定为2）
        padded_vector = torch.zeros(self.max_actions, 2)
        padded_label = torch.zeros(self.max_actions, 1)
        
        # 填充真实数据
        padded_vector[:n_actions] = vector
        padded_label[:n_actions] = label
        
        return {
            'image': image,
            'vector': padded_vector,
            'label': padded_label,
            'n_actions': n_actions
        }
    
    @staticmethod
    def collate_fn(batch):
        """自定义批次整理函数"""
        images = torch.stack([item['image'] for item in batch])
        vectors = torch.stack([item['vector'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        n_actions = torch.tensor([item['n_actions'] for item in batch])
        
        return {
            'images': images,
            'vectors': vectors,
            'labels': labels,
            'n_actions': n_actions
        }


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列中的每个位置添加位置信息。
    """
    
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 形状为 (seq_len, batch_size, d_model) 的张量
        """
        return x + self.pe[:x.size(0), :]


class PolicyUNet(nn.Module):
    """
    基于UNet的策略模型，接受图片和n×2向量作为输入，
    输出n×1的概率值用于动作排序。
    
    参数:
        n_channels (int): 输入图片的通道数 (例如：RGB为3，灰度图为1)
        bilinear (bool): 是否使用双线性上采样或转置卷积
        feature_fusion_method (str): 融合图像和向量特征的方法 ('concat', 'add', 'attention')
    """
    
    def __init__(self, n_channels=3, bilinear=False, feature_fusion_method='concat'):
        super(PolicyUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.feature_fusion_method = feature_fusion_method
        
        # 用于图像特征的UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 向量特征处理（固定输入维度为2）
        self.vector_processor = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合层
        if feature_fusion_method == 'concat':
            # 如果使用拼接，需要调整解码器的输入通道数
            fusion_channels = 1024 // factor + 512  # UNet瓶颈层 + 向量特征
        elif feature_fusion_method == 'add':
            fusion_channels = 1024 // factor
            # 向量特征将被投影以匹配UNet瓶颈层的维度
            self.vector_projection = nn.Linear(512, 1024 // factor)
        elif feature_fusion_method == 'attention':
            fusion_channels = 1024 // factor
            # 使用注意力机制融合向量和图像特征
            self.attention = CrossModalAttention(1024 // factor, 512)
        else:
            raise ValueError(f"未知的融合方法: {feature_fusion_method}")
        
        # 修改的UNet解码器，用于处理融合特征
        self.fusion_conv = DoubleConv(fusion_channels, 512 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 全局平均池化和最终分类层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 每个动作输出单个概率值
        )
        
    def forward(self, image, vector):
        """
        策略模型的前向传播。
        
        参数:
            image (torch.Tensor): 输入图像，形状为 (batch_size, n_channels, H, W)
            vector (torch.Tensor): 输入向量，形状为 (batch_size, n, 2)，其中n是动作数量
            
        返回:
            torch.Tensor: 输出概率，形状为 (batch_size, n, 1)
        """
        batch_size, n_actions, _ = vector.shape  # 向量维度固定为2
        
        # 通过UNet编码器处理图像
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # 瓶颈层特征: (batch_size, channels, H', W')
        
        # 分别处理每个动作向量
        vector_reshaped = vector.view(batch_size * n_actions, 2)
        vector_features = self.vector_processor(vector_reshaped)  # (batch_size * n_actions, 512)
        
        # 为融合准备 - 为每个动作重复图像特征
        # x5 形状: (batch_size, channels, H', W')
        h, w = x5.shape[2], x5.shape[3]
        image_features_repeated = x5.unsqueeze(1).repeat(1, n_actions, 1, 1, 1)
        image_features_flat = image_features_repeated.view(batch_size * n_actions, -1, h, w)
        
        # 特征融合
        if self.feature_fusion_method == 'concat':
            # 向量特征的空间广播并拼接
            vector_spatial = vector_features.view(batch_size * n_actions, -1, 1, 1)
            vector_spatial = vector_spatial.expand(-1, -1, h, w)
            fused_features = torch.cat([image_features_flat, vector_spatial], dim=1)
            
        elif self.feature_fusion_method == 'add':
            # 投影向量特征以匹配图像特征维度
            vector_projected = self.vector_projection(vector_features)
            vector_spatial = vector_projected.view(batch_size * n_actions, -1, 1, 1)
            vector_spatial = vector_spatial.expand(-1, -1, h, w)
            fused_features = image_features_flat + vector_spatial
            
        elif self.feature_fusion_method == 'attention':
            # 使用注意力机制进行融合
            fused_features = self.attention(image_features_flat, vector_features, h, w)
        
        # 通过融合卷积处理融合特征
        fused_features = self.fusion_conv(fused_features)
        
        # 为UNet解码器重塑形状（需要处理跳跃连接）
        # 为了简化，我们将融合特征作为解码器的输入
        x = self.up1(fused_features.view(batch_size * n_actions, -1, h, w), 
                     x4.unsqueeze(1).repeat(1, n_actions, 1, 1, 1).view(batch_size * n_actions, -1, x4.shape[2], x4.shape[3]))
        x = self.up2(x, x3.unsqueeze(1).repeat(1, n_actions, 1, 1, 1).view(batch_size * n_actions, -1, x3.shape[2], x3.shape[3]))
        x = self.up3(x, x2.unsqueeze(1).repeat(1, n_actions, 1, 1, 1).view(batch_size * n_actions, -1, x2.shape[2], x2.shape[3]))
        x = self.up4(x, x1.unsqueeze(1).repeat(1, n_actions, 1, 1, 1).view(batch_size * n_actions, -1, x1.shape[2], x1.shape[3]))
        
        # 全局池化和分类
        x = self.global_pool(x)  # (batch_size * n_actions, 64, 1, 1)
        x = x.view(batch_size * n_actions, -1)  # (batch_size * n_actions, 64)
        
        # 最终分类
        probabilities = self.classifier(x)  # (batch_size * n_actions, 1)
        probabilities = torch.sigmoid(probabilities)  # 应用sigmoid获得概率输出
        
        # 重塑为 (batch_size, n_actions, 1)
        probabilities = probabilities.view(batch_size, n_actions, 1)
        
        return probabilities


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制，用于融合图像和向量特征。
    """
    
    def __init__(self, image_dim, vector_dim):
        super(CrossModalAttention, self).__init__()
        self.image_dim = image_dim
        self.vector_dim = vector_dim
        
        # 注意力层
        self.query_proj = nn.Linear(vector_dim, image_dim)
        self.key_proj = nn.Conv2d(image_dim, image_dim, 1)
        self.value_proj = nn.Conv2d(image_dim, image_dim, 1)
        
        self.scale = image_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, image_features, vector_features, h, w):
        """
        参数:
            image_features: (batch_size * n_actions, image_dim, h, w)
            vector_features: (batch_size * n_actions, vector_dim)
            h, w: 空间维度
        
        返回:
            fused_features: (batch_size * n_actions, image_dim, h, w)
        """
        batch_n = image_features.shape[0]
        
        # 将向量特征投影为查询
        query = self.query_proj(vector_features)  # (batch_size * n_actions, image_dim)
        query = query.unsqueeze(-1).unsqueeze(-1)  # (batch_size * n_actions, image_dim, 1, 1)
        
        # 将图像特征投影为键和值
        key = self.key_proj(image_features)  # (batch_size * n_actions, image_dim, h, w)
        value = self.value_proj(image_features)  # (batch_size * n_actions, image_dim, h, w)
        
        # 为注意力计算重塑形状
        key = key.view(batch_n, self.image_dim, -1)  # (batch_size * n_actions, image_dim, h*w)
        value = value.view(batch_n, self.image_dim, -1)  # (batch_size * n_actions, image_dim, h*w)
        query = query.view(batch_n, self.image_dim, 1)  # (batch_size * n_actions, image_dim, 1)
        
        # 计算注意力权重
        attention_scores = torch.bmm(query.transpose(1, 2), key) * self.scale  # (batch_size * n_actions, 1, h*w)
        attention_weights = self.softmax(attention_scores)  # (batch_size * n_actions, 1, h*w)
        
        # 将注意力应用于值
        attended = torch.bmm(value, attention_weights.transpose(1, 2))  # (batch_size * n_actions, image_dim, 1)
        attended = attended.view(batch_n, self.image_dim, 1, 1)  # (batch_size * n_actions, image_dim, 1, 1)
        attended = attended.expand(-1, -1, h, w)  # (batch_size * n_actions, image_dim, h, w)
        
        # 与原始图像特征组合
        fused_features = image_features + attended
        
        return fused_features


class AdaptivePolicyUNet(nn.Module):
    """
    自适应PolicyUNet，支持动态的n（动作数量）。
    向量维度固定为2，使用注意力机制处理可变长度序列。
    """
    
    def __init__(self, n_channels=3, hidden_dim=256, max_sequence_length=100, use_positional_encoding=True):
        super(AdaptivePolicyUNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding
        
        # 图像编码器（简化的UNet）
        self.image_encoder = nn.Sequential(
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            nn.AdaptiveAvgPool2d(1)  # 全局池化
        )
        
        # 向量编码器（输入维度固定为2）
        self.vector_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 位置编码（如果启用）
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_dim, max_sequence_length)
        
        # 自注意力机制处理可变长度序列
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 跨模态注意力融合图像和序列特征
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, vector):
        """
        自适应前向传播，支持动态的n（动作数量）。
        
        参数:
            image (torch.Tensor): 输入图像，形状为 (batch_size, n_channels, H, W)
            vector (torch.Tensor): 输入向量，形状为 (batch_size, n, 2)
            
        返回:
            torch.Tensor: 输出概率，形状为 (batch_size, n, 1)
        """
        batch_size, n_actions, _ = vector.shape  # 向量维度固定为2
        
        # 编码图像
        image_features = self.image_encoder(image)  # (batch_size, 512, 1, 1)
        image_features = image_features.view(batch_size, -1)  # (batch_size, 512)
        
        # 将图像特征投影到hidden_dim
        image_projected = nn.Linear(512, self.hidden_dim).to(image.device)(image_features)
        image_projected = image_projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 处理向量
        vector_reshaped = vector.view(batch_size * n_actions, 2)
        vector_features = self.vector_encoder(vector_reshaped)  # (batch_size * n_actions, hidden_dim)
        vector_features = vector_features.view(batch_size, n_actions, self.hidden_dim)
        
        # 添加位置编码（如果启用）
        if self.use_positional_encoding:
            # 转换为 (seq_len, batch_size, hidden_dim) 格式
            vector_for_pe = vector_features.transpose(0, 1)  # (n_actions, batch_size, hidden_dim)
            vector_for_pe = self.positional_encoding(vector_for_pe)
            vector_features = vector_for_pe.transpose(0, 1)  # (batch_size, n_actions, hidden_dim)
        
        # 自注意力处理向量序列
        action_mask = torch.ones(batch_size, n_actions, device=vector.device, dtype=torch.bool)
        
        vector_attended, _ = self.self_attention(
            vector_features, vector_features, vector_features,
            key_padding_mask=~action_mask
        )
        
        # 跨模态注意力：使用图像特征作为查询，向量特征作为键和值
        cross_attended, attention_weights = self.cross_attention(
            image_projected.expand(-1, n_actions, -1),  # 为每个动作复制图像特征
            vector_attended,
            vector_attended,
            key_padding_mask=~action_mask
        )
        
        # 最终分类
        probabilities = self.classifier(cross_attended)  # (batch_size, n_actions, 1)
        
        return probabilities


class SimplePolicyUNet(nn.Module):
    """
    PolicyUNet的简化版本，用于更快的推理。
    向量维度固定为2，使用更直接的架构。
    """
    
    def __init__(self, n_channels=3, hidden_dim=256):
        super(SimplePolicyUNet, self).__init__()
        self.n_channels = n_channels
        
        # 图像编码器（简化的UNet）
        self.image_encoder = nn.Sequential(
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            nn.AdaptiveAvgPool2d(1)  # 全局池化
        )
        
        # 向量编码器（输入维度固定为2）
        self.vector_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合和分类
        self.classifier = nn.Sequential(
            nn.Linear(512 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, vector):
        """
        参数:
            image (torch.Tensor): 输入图像，形状为 (batch_size, n_channels, H, W)
            vector (torch.Tensor): 输入向量，形状为 (batch_size, n, 2)
        
        返回:
            torch.Tensor: 输出概率，形状为 (batch_size, n, 1)
        """
        batch_size, n_actions, _ = vector.shape  # 向量维度固定为2
        
        # 编码图像
        image_features = self.image_encoder(image)  # (batch_size, 512, 1, 1)
        image_features = image_features.view(batch_size, -1)  # (batch_size, 512)
        
        # 为每个动作编码向量
        vector_reshaped = vector.view(batch_size * n_actions, 2)
        vector_features = self.vector_encoder(vector_reshaped)  # (batch_size * n_actions, hidden_dim)
        
        # 为每个动作重复图像特征
        image_features_repeated = image_features.unsqueeze(1).repeat(1, n_actions, 1)
        image_features_flat = image_features_repeated.view(batch_size * n_actions, -1)
        
        # 拼接特征并分类
        combined_features = torch.cat([image_features_flat, vector_features], dim=1)
        probabilities = self.classifier(combined_features)  # (batch_size * n_actions, 1)
        
        # 重塑为 (batch_size, n_actions, 1)
        probabilities = probabilities.view(batch_size, n_actions, 1)
        
        return probabilities


class FlexiblePolicyUNet(nn.Module):
    """
    灵活的策略模型，使用Transformer架构来处理任意长度的动作序列。
    向量维度固定为2，支持不同的动作数量n。
    """
    
    def __init__(self, n_channels=3, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        super(FlexiblePolicyUNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 投影层
        self.image_projection = nn.Linear(512, hidden_dim)
        self.vector_projection = nn.Linear(2, hidden_dim)  # 向量维度固定为2
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def create_padding_mask(self, x, lengths):
        """创建填充掩码"""
        batch_size, max_len = x.shape[:2]
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask
        
    def forward(self, image, vector):
        """
        参数:
            image: (batch_size, n_channels, H, W)
            vector: (batch_size, max_n, 2) - 向量维度固定为2
        
        注意：这个版本自动检测有效的向量长度（非零向量）
        """
        batch_size, max_n, _ = vector.shape  # 向量维度固定为2
        
        # 自动检测每个样本的有效向量长度（基于非零向量）
        vector_lengths = []
        for i in range(batch_size):
            # 计算非零向量的数量
            non_zero_mask = (vector[i].abs().sum(dim=1) > 1e-6)
            length = non_zero_mask.sum().item()
            vector_lengths.append(max(1, length))  # 至少保证长度为1
        vector_lengths = torch.tensor(vector_lengths, device=vector.device)
        
        # 编码图像
        image_features = self.image_encoder(image).view(batch_size, -1)
        image_features = self.image_projection(image_features)  # (batch_size, hidden_dim)
        
        # 投影向量
        vector_features = self.vector_projection(vector)  # (batch_size, max_n, hidden_dim)
        
        # 将图像特征添加到每个动作向量
        image_features_expanded = image_features.unsqueeze(1).expand(-1, max_n, -1)
        combined_features = vector_features + image_features_expanded
        
        # 创建填充掩码
        padding_mask = self.create_padding_mask(combined_features, vector_lengths)
            
        # Transformer编码
        encoded = self.transformer_encoder(combined_features, src_key_padding_mask=padding_mask)
        
        # 输出预测
        probabilities = self.output_projection(encoded)  # (batch_size, max_n, 1)
        
        return probabilities


def create_policy_model(model_type='simple', **kwargs):
    """
    创建策略模型的工厂函数。
    
    参数:
        model_type (str): 模型类型 ('simple', 'full', 'adaptive', 'flexible')
        **kwargs: 模型初始化的额外参数（注意：向量维度固定为2）
    
    返回:
        nn.Module: 策略模型实例
    """
    if model_type == 'simple':
        return SimplePolicyUNet(**kwargs)
    elif model_type == 'full':
        return PolicyUNet(**kwargs)
    elif model_type == 'adaptive':
        return AdaptivePolicyUNet(**kwargs)
    elif model_type == 'flexible':
        return FlexiblePolicyUNet(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def create_mixed_batch_data(batch_size, n_channels, image_size, max_actions):
    """
    创建用于测试的混合批次数据（不同样本有不同数量的动作）
    
    Args:
        batch_size: 批次大小
        n_channels: 图像通道数
        image_size: 图像尺寸
        max_actions: 最大动作数量
    
    Returns:
        tuple: (图像数据, 填充后的向量数据, 向量长度)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建图像数据
    image = torch.randn(batch_size, n_channels, image_size, image_size).to(device)
    
    # 为每个样本随机生成不同的动作数量
    vector_lengths = torch.randint(1, max_actions + 1, (batch_size,))
    
    # 创建填充后的向量数据（向量维度固定为2）
    vector = torch.zeros(batch_size, max_actions, 2).to(device)
    
    for i in range(batch_size):
        # 为第i个样本生成实际长度的向量
        actual_length = vector_lengths[i].item()
        vector[i, :actual_length] = torch.randn(actual_length, 2).to(device)
    
    return image, vector, vector_lengths.to(device)


def train_adaptive_model_example():
    """
    展示如何训练自适应模型以处理不同的n（动作数量）。
    向量维度固定为2。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟数据集，包含不同的动作数量
    def create_sample_data():
        images = []
        vectors_list = []
        labels_list = []
        
        # 生成不同动作数量的样本
        action_counts = [3, 5, 8, 10, 12]  # 不同的动作数量
        
        for _ in range(100):  # 每种配置生成100个样本
            for n_actions in action_counts:
                # 生成图像
                image = torch.randn(3, 256, 256)
                images.append(image)
                
                # 生成向量和标签（向量维度固定为2）
                vector = torch.randn(n_actions, 2)  # 可能代表坐标点
                label = torch.rand(n_actions, 1)  # 随机概率作为标签
                
                vectors_list.append(vector)
                labels_list.append(label)
        
        return images, vectors_list, labels_list
    
    print("创建混合动作数量数据集...")
    images, vectors_list, labels_list = create_sample_data()
    dataset = MixedDimensionDataset(images, vectors_list, labels_list)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=MixedDimensionDataset.collate_fn
    )
    
    # 创建自适应模型
    model = FlexiblePolicyUNet(
        n_channels=3,
        hidden_dim=256
    ).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("开始训练...")
    model.train()
    
    # 训练一个epoch作为示例
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        images = batch['images'].to(device)
        vectors = batch['vectors'].to(device)
        labels = batch['labels'].to(device)
        n_actions = batch['n_actions'].to(device)
        
        # 前向传播（现在所有模型都使用相同的forward签名）
        predictions = model(images, vectors)
        
        # 计算损失（只对有效位置计算）
        loss = 0
        for i, n in enumerate(n_actions):
            valid_pred = predictions[i, :n, :]
            valid_label = labels[i, :n, :]
            loss += criterion(valid_pred, valid_label)
        
        loss = loss / len(n_actions)  # 平均损失
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"批次 {num_batches}, 损失: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"平均训练损失: {avg_loss:.4f}")
    
    return model


def create_mixed_batch_data(batch_size, n_channels, image_size, max_actions):
    """
    创建混合批次数据，包含不同的n（动作数量）。
    向量维度固定为2。
    
    参数:
        batch_size: 批次大小
        n_channels: 图像通道数
        image_size: 图像尺寸
        max_actions: 最大动作数量
        
    返回:
        image, padded_vector, vector_lengths
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建图像
    image = torch.randn(batch_size, n_channels, image_size, image_size).to(device)
    
    # 为每个样本随机生成不同的动作数量
    vector_lengths = torch.randint(1, max_actions + 1, (batch_size,))
    
    # 创建填充的向量张量（向量维度固定为2）
    padded_vector = torch.zeros(batch_size, max_actions, 2).to(device)
    
    for i in range(batch_size):
        n_actions = vector_lengths[i].item()
        
        # 为每个样本填充真实数据（可能代表坐标点）
        padded_vector[i, :n_actions, :] = torch.randn(n_actions, 2).to(device)
    
    return image, padded_vector, vector_lengths.to(device)
    """
    创建混合批次数据，包含不同的n和m组合。
    
    参数:
        batch_size: 批次大小
        n_channels: 图像通道数
        image_size: 图像尺寸
        max_actions: 最大动作数量
        max_vector_dim: 最大向量维度
        
    返回:
        image, padded_vector, vector_lengths, actual_dims
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建图像
    image = torch.randn(batch_size, n_channels, image_size, image_size).to(device)
    
    # 为每个样本随机生成不同的n和m
    vector_lengths = torch.randint(1, max_actions + 1, (batch_size,))
    actual_dims = torch.randint(32, max_vector_dim + 1, (batch_size,))
    
    # 创建填充的向量张量
    padded_vector = torch.zeros(batch_size, max_actions, max_vector_dim).to(device)
    
    for i in range(batch_size):
        n_actions = vector_lengths[i].item()
        m_features = actual_dims[i].item()
        
        # 为每个样本填充真实数据
        padded_vector[i, :n_actions, :m_features] = torch.randn(n_actions, m_features).to(device)
    
    return image, padded_vector, vector_lengths.to(device), actual_dims


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    batch_size = 2
    n_channels = 3
    image_size = 256
    max_actions = 10
    
    print("=== 测试不同模型架构（向量维度固定为2）===")
    
    # 测试简单模型（固定维度）
    print("\n1. 测试SimplePolicyUNet（向量维度固定为2）...")
    n_actions = 5
    
    image = torch.randn(batch_size, n_channels, image_size, image_size).to(device)
    vector = torch.randn(batch_size, n_actions, 2).to(device)  # 向量维度固定为2
    
    simple_model = SimplePolicyUNet(n_channels=n_channels).to(device)
    simple_output = simple_model(image, vector)
    print(f"简单模型输出形状: {simple_output.shape}")
    print(f"简单模型参数数量: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    # 测试自适应模型
    print("\n2. 测试AdaptivePolicyUNet（自适应动作数量）...")
    adaptive_model = AdaptivePolicyUNet(
        n_channels=n_channels, 
        hidden_dim=256
    ).to(device)
    
    # 测试不同的动作数量
    test_cases = [3, 7, 10]  # 不同的动作数量
    
    for n in test_cases:
        test_vector = torch.randn(batch_size, n, 2).to(device)
        adaptive_output = adaptive_model(image, test_vector)
        print(f"自适应模型 (n={n}) 输出形状: {adaptive_output.shape}")
    
    print(f"自适应模型参数数量: {sum(p.numel() for p in adaptive_model.parameters()):,}")
    
    # 测试灵活模型（支持批次内不同动作数量）
    print("\n3. 测试FlexiblePolicyUNet（批次内混合动作数量）...")
    flexible_model = FlexiblePolicyUNet(
        n_channels=n_channels,
        hidden_dim=256
    ).to(device)
    
    # 创建混合批次数据
    mixed_image, mixed_vector, vector_lengths = create_mixed_batch_data(
        batch_size, n_channels, image_size, max_actions
    )
    
    # 现在所有模型都使用相同的forward签名
    flexible_output = flexible_model(mixed_image, mixed_vector)
    print(f"灵活模型输出形状: {flexible_output.shape}")
    print(f"灵活模型参数数量: {sum(p.numel() for p in flexible_model.parameters()):,}")
    
    print(f"批次中实际动作数量: {vector_lengths.tolist()}")
    
    # 测试完整模型
    print("\n4. 测试PolicyUNet（完整版）...")
    full_model = PolicyUNet(n_channels=n_channels).to(device)
    full_output = full_model(image, vector)
    print(f"完整模型输出形状: {full_output.shape}")
    print(f"完整模型参数数量: {sum(p.numel() for p in full_model.parameters()):,}")
    
    print("\n=== 所有测试通过！ ===")
    print("\n模型选择建议:")
    print("- SimplePolicyUNet: 固定动作数量n，推理速度最快")
    print("- AdaptivePolicyUNet: 支持不同动作数量n，使用注意力机制")
    print("- FlexiblePolicyUNet: 支持批次内混合动作数量，最灵活但计算开销较大")
    print("- PolicyUNet: 完整UNet架构，特征表达能力最强")
    print("\n注意：所有模型的向量维度都固定为2（可能代表坐标点等）")
    
    # 可选：运行训练示例
    run_training_example = input("\n是否运行混合动作数量训练示例？(y/n): ").lower().strip() == 'y'
    if run_training_example:
        print("\n=== 运行混合动作数量训练示例 ===")
        trained_model = train_adaptive_model_example()
        print("训练示例完成！")
