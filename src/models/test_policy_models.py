"""
简单测试脚本，用于验证策略模型的调用是否正确。
验证所有模型都使用统一的forward(image, vector)签名。
"""

import torch
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.policy_model import create_policy_model, FlexiblePolicyUNet

def test_model_forward():
    """测试所有模型的forward方法使用统一签名"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    n_points = 5
    
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    points = torch.randn(batch_size, n_points, 2).to(device)
    
    print("测试数据形状:")
    print(f"  images: {images.shape}")
    print(f"  points: {points.shape}")
    
    # 测试所有模型类型
    model_types = ['simple', 'adaptive', 'full', 'flexible']
    
    print("\n=== 测试所有模型使用统一的forward(image, vector)签名 ===")
    
    for model_type in model_types:
        print(f"\n测试模型类型: {model_type}")
        
        try:
            if model_type == 'flexible':
                model = FlexiblePolicyUNet(n_channels=3, hidden_dim=256).to(device)
            else:
                model = create_policy_model(
                    model_type=model_type, 
                    n_channels=3, 
                    hidden_dim=256
                ).to(device)
            
            print(f"  模型类: {model.__class__.__name__}")
            print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 测试forward方法 - 所有模型现在都使用相同的签名
            with torch.no_grad():
                output = model(images, points)
                print(f"  输出形状: {output.shape}")
                print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
                
            print(f"  ✓ {model_type} 模型测试成功")
            
        except Exception as e:
            print(f"  ✗ {model_type} 模型测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试不同尺寸的输入
    print("\n=== 测试不同输入尺寸 ===")
    
    test_cases = [
        (1, 3, 2),   # 小批次，少量点
        (4, 10, 2),  # 中等批次，中等点数
        (2, 20, 2),  # 少批次，大量点
    ]
    
    # 使用flexible模型测试不同尺寸
    model = FlexiblePolicyUNet(n_channels=3, hidden_dim=128).to(device)
    
    for batch_sz, n_pts, vec_dim in test_cases:
        print(f"\n测试尺寸: batch={batch_sz}, points={n_pts}, dim={vec_dim}")
        try:
            test_images = torch.randn(batch_sz, 3, 256, 256).to(device)
            test_points = torch.randn(batch_sz, n_pts, vec_dim).to(device)
            
            with torch.no_grad():
                output = model(test_images, test_points)
                print(f"  输出形状: {output.shape}")
                print(f"  ✓ 尺寸测试成功")
                
        except Exception as e:
            print(f"  ✗ 尺寸测试失败: {e}")
    
    # 测试FlexiblePolicyUNet的自动长度检测
    print("\n=== 测试FlexiblePolicyUNet自动长度检测 ===")
    
    # 创建带有不同有效长度的测试数据
    batch_size = 3
    max_points = 8
    test_images = torch.randn(batch_size, 3, 256, 256).to(device)
    test_points = torch.zeros(batch_size, max_points, 2).to(device)
    
    # 为每个样本设置不同的有效点数
    valid_lengths = [3, 5, 7]
    for i, length in enumerate(valid_lengths):
        test_points[i, :length, :] = torch.randn(length, 2).to(device)
    
    print(f"手动设置的有效长度: {valid_lengths}")
    
    try:
        with torch.no_grad():
            output = model(test_images, test_points)
            print(f"FlexiblePolicyUNet输出形状: {output.shape}")
            
            # 检查每个样本的非零输出
            for i in range(batch_size):
                non_zero_outputs = (output[i].abs() > 1e-6).sum().item()
                print(f"  样本{i}: 非零输出数量={non_zero_outputs}, 预期有效长度={valid_lengths[i]}")
            
            print("  ✓ 自动长度检测测试成功")
            
    except Exception as e:
        print(f"  ✗ 自动长度检测测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_forward()
