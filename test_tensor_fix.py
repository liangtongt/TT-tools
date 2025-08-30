#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Tensor处理修复的脚本
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensor_handling():
    """测试Tensor处理是否正确"""
    
    try:
        from image_sequence_compressor import ImageSequenceCompressor
        
        # 创建测试数据
        # 模拟ComfyUI的Tensor输入
        test_images = [
            torch.rand(3, 256, 256),  # 0-1范围的Tensor
            torch.rand(3, 256, 256) * 255,  # 0-255范围的Tensor
        ]
        
        test_image = torch.rand(3, 512, 512)  # 容器图像
        
        # 创建节点实例
        compressor = ImageSequenceCompressor()
        
        print("测试Tensor处理...")
        print(f"输入images类型: {type(test_images[0])}")
        print(f"输入image类型: {type(test_image)}")
        print(f"Tensor形状: {test_images[0].shape}")
        print(f"Tensor数值范围: {test_images[0].min():.3f} - {test_images[0].max():.3f}")
        
        # 调用压缩函数
        result = compressor.compress_sequence(
            images=test_images,
            image=test_image,
            compression_level=6,
            quality=95,
            format="PNG",
            include_metadata=True
        )
        
        print(f"输出类型: {type(result[0])}")
        print(f"输出形状: {result[0].shape}")
        print(f"输出数值范围: {result[0].min():.3f} - {result[0].max():.3f}")
        
        print("✅ Tensor处理测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ Tensor处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Tensor处理修复测试 ===\n")
    
    success = test_tensor_handling()
    
    if success:
        print("\n🎉 测试通过！Tensor处理修复成功。")
    else:
        print("\n💥 测试失败，需要进一步修复。")
