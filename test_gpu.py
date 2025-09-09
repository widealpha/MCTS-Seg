#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from pathlib import Path

def test_gpu():
    print("Testing GPU availability...")
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Test GPU tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    else:
        print("✗ CUDA is not available")
    
    # Test basic imports
    try:
        from src.utils.helpers import calculate_iou, calculate_dice
        print("✓ Utils import successful")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
    
    # Test file paths
    baseline_dir = "/home/kmh/ai/MCTS-Seg/results/baseline/ISIC2016/medical_sam_adapter/center_1"
    mcts_dir = "/home/kmh/ai/MCTS-Seg/results/mcts/ISIC2016/m2g6s512a1bg0t0rt0f1/msa-2025-06-29_14-24-03"
    gt_dir = "/home/kmh/ai/MCTS-Seg/data/ISIC2016/raw/test/gt"
    
    for name, path in [("Baseline", baseline_dir), ("MCTS", mcts_dir), ("GT", gt_dir)]:
        if Path(path).exists():
            file_count = len(list(Path(path).glob("*.png")))
            print(f"✓ {name} directory exists with {file_count} PNG files")
        else:
            print(f"✗ {name} directory does not exist: {path}")

if __name__ == "__main__":
    test_gpu()
