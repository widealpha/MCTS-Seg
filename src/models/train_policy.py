"""
策略模型训练脚本。
基于UNet架构的MCTS分割策略模型训练。
"""

import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np

# 导入策略模型和数据加载器
from policy_model import create_policy_model, FlexiblePolicyUNet
from src.data.policy_loader import get_policy_data_loader
from src.utils.helpers import get_log_writer, setup_seed, get_checkpoints_path
from src.cfg import parse_args

# 解析参数
device = parse_args().device
dataset = parse_args().dataset

# 创建检查点目录
checkpoints_path = os.path.join(
    get_checkpoints_path(), 
    'policy_model',
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(checkpoints_path, exist_ok=True)


def weighted_binary_cross_entropy_loss(predictions, targets, point_lengths, pos_weight=2.0):
    """
    加权二元交叉熵损失，处理不平衡的正负样本。
    
    Args:
        predictions: 模型预测值 (batch_size, max_points, 1)
        targets: 真实标签 (batch_size, max_points, 1)
        point_lengths: 每个样本的实际点数 (batch_size,)
        pos_weight: 正样本权重
    
    Returns:
        平均损失值
    """
    batch_size, max_points, _ = predictions.shape
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(predictions.device))
    
    for i in range(batch_size):
        n_points = point_lengths[i].item()
        if n_points > 0:
            pred_i = predictions[i, :n_points, :]  # (n_points, 1)
            target_i = targets[i, :n_points, :]   # (n_points, 1)
            
            loss_i = criterion(pred_i, target_i)
            total_loss += loss_i * n_points
            total_samples += n_points
    
    return total_loss / max(total_samples, 1)


def compute_metrics(predictions, targets, point_lengths, threshold=0.5):
    """
    计算分类指标。
    
    Args:
        predictions: 模型预测值 (batch_size, max_points, 1)
        targets: 真实标签 (batch_size, max_points, 1)
        point_lengths: 每个样本的实际点数 (batch_size,)
        threshold: 分类阈值
    
    Returns:
        dict: 包含accuracy, precision, recall, f1的字典
    """
    batch_size = predictions.shape[0]
    
    all_preds = []
    all_targets = []
    
    # 应用sigmoid激活函数
    predictions = torch.sigmoid(predictions)
    
    for i in range(batch_size):
        n_points = point_lengths[i].item()
        if n_points > 0:
            pred_i = predictions[i, :n_points, 0].cpu().numpy()  # (n_points,)
            target_i = targets[i, :n_points, 0].cpu().numpy()   # (n_points,)
            
            all_preds.extend(pred_i)
            all_targets.extend(target_i)
    
    if len(all_preds) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 二分类预测
    pred_binary = (all_preds > threshold).astype(int)
    target_binary = (all_targets > threshold).astype(int)
    
    # 计算指标
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_policy_model(old_checkpoint=None, model_type='flexible'):
    """
    训练策略模型。
    
    Args:
        old_checkpoint: 预训练模型路径
        model_type: 模型类型 ('simple', 'adaptive', 'flexible', 'full')
    """
    print(f"开始训练策略模型: {model_type}")
    print(f"数据集: {dataset}")
    print(f"设备: {device}")
    
    # 获取数据加载器
    train_dataloader, test_dataloader = get_policy_data_loader(
        batch_size=8,  # 较小的batch size以适应不同长度的序列
        shuffle=True,
        test_batch_size=8,
        test_shuffle=False,
        min_points=5,
        max_points=20,
        image_size=256,
        num_workers=4
    )
    
    print(f"训练集批次数: {len(train_dataloader)}")
    print(f"测试集批次数: {len(test_dataloader)}")
    
    # 获取样本形状信息
    first_batch = next(iter(train_dataloader))
    if first_batch is None:
        raise ValueError("无法获取有效的训练数据")
    
    sample_image_shape = first_batch['images'][0].shape
    print(f"样本图像形状: {sample_image_shape}")
    
    # 训练参数
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 100
    patience = 20
    pos_weight = 2.0  # 正样本权重，处理类别不平衡
    
    # 初始化模型
    if model_type == 'flexible':
        model = FlexiblePolicyUNet(
            n_channels=3,
            hidden_dim=256
        ).to(device)
    else:
        model = create_policy_model(
            model_type=model_type,
            n_channels=3,
            hidden_dim=256
        ).to(device)
    
    # 加载预训练模型
    if old_checkpoint and os.path.exists(old_checkpoint):
        model.load_state_dict(torch.load(old_checkpoint, map_location=device))
        print(f"加载预训练模型: {old_checkpoint}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 早停机制
    best_test_loss = float('inf')
    patience_counter = 0
    
    # 日志记录器
    log_writer = get_log_writer(log_path=os.path.join(checkpoints_path, 'logs'))
    
    print("开始训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0
        train_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            if batch is None:
                continue
                
            images = batch['images'].to(device)
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            point_lengths = batch['point_lengths'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device):
                # 前向传播（所有模型现在都使用相同的签名）
                predictions = model(images, points)
                
                # 计算损失
                loss = weighted_binary_cross_entropy_loss(
                    predictions, labels, point_lengths, pos_weight
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_steps += 1
            
            # 计算指标
            with torch.no_grad():
                batch_metrics = compute_metrics(predictions, labels, point_lengths)
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]
        
        # 平均训练指标
        avg_train_loss = train_loss / max(train_steps, 1)
        for key in train_metrics:
            train_metrics[key] /= max(train_steps, 1)
        
        # 记录训练指标
        log_writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        log_writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
        log_writer.add_scalar('Train/Precision', train_metrics['precision'], epoch)
        log_writer.add_scalar('Train/Recall', train_metrics['recall'], epoch)
        log_writer.add_scalar('Train/F1', train_metrics['f1'], epoch)
        log_writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  训练指标 - Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        # 测试阶段
        test_loss, test_metrics = evaluate_policy_model(
            model, test_dataloader, pos_weight, device, model_type
        )
        
        # 记录测试指标
        log_writer.add_scalar('Test/Loss', test_loss, epoch)
        log_writer.add_scalar('Test/Accuracy', test_metrics['accuracy'], epoch)
        log_writer.add_scalar('Test/Precision', test_metrics['precision'], epoch)
        log_writer.add_scalar('Test/Recall', test_metrics['recall'], epoch)
        log_writer.add_scalar('Test/F1', test_metrics['f1'], epoch)
        
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  测试指标 - Acc: {test_metrics['accuracy']:.4f}, "
              f"Prec: {test_metrics['precision']:.4f}, "
              f"Rec: {test_metrics['recall']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}")
        
        # 学习率调度
        scheduler.step(test_loss)
        
        # 早停和模型保存
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(checkpoints_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存最佳模型: {best_model_path}")
        else:
            patience_counter += 1
            print(f"  早停计数器: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("触发早停！")
                break
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoints_path, 
                f'epoch_{epoch}_{datetime.now().strftime("%H-%M-%S")}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoints_path, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 保存训练配置
    config_info = {
        'model_type': model_type,
        'lr': lr,
        'weight_decay': weight_decay,
        'pos_weight': pos_weight,
        'epochs': epoch + 1,
        'best_test_loss': best_test_loss,
        'final_test_loss': test_loss
    }
    
    for key, value in config_info.items():
        log_writer.add_text('Config', f'{key}: {value}')
    
    log_writer.close()
    print(f"训练完成！最佳测试损失: {best_test_loss:.4f}")
    print(f"模型保存在: {checkpoints_path}")
    
    return model


def evaluate_policy_model(model, test_dataloader, pos_weight, device, model_type):
    """
    评估策略模型。
    
    Args:
        model: 要评估的模型
        test_dataloader: 测试数据加载器
        pos_weight: 正样本权重
        device: 计算设备
        model_type: 模型类型
    
    Returns:
        tuple: (测试损失, 测试指标字典)
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='测试中'):
            if batch is None:
                continue
                
            images = batch['images'].to(device)
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            point_lengths = batch['point_lengths'].to(device)
            
            # 前向传播（所有模型现在都使用相同的签名）
            predictions = model(images, points)
            
            # 计算损失
            loss = weighted_binary_cross_entropy_loss(
                predictions, labels, point_lengths, pos_weight
            )
            
            total_loss += loss.item()
            total_steps += 1
            
            # 计算指标
            batch_metrics = compute_metrics(predictions, labels, point_lengths)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]
    
    # 平均指标
    avg_test_loss = total_loss / max(total_steps, 1)
    for key in all_metrics:
        all_metrics[key] /= max(total_steps, 1)
    
    return avg_test_loss, all_metrics


if __name__ == '__main__':
    # 设置随机种子
    setup_seed()
    
    # 训练不同类型的模型
    model_types = ['flexible']  # 可以尝试其他类型: ['simple', 'adaptive', 'flexible', 'full']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"训练模型类型: {model_type}")
        print(f"{'='*50}")
        
        try:
            trained_model = train_policy_model(
                old_checkpoint=None,  # 可以指定预训练模型路径
                model_type=model_type
            )
            print(f"模型 {model_type} 训练完成！")
        except Exception as e:
            print(f"训练模型 {model_type} 时出错: {e}")
            continue
    
    print("\n所有模型训练完成！")