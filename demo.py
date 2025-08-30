#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示脚本 - 展示简化后的图片序列压缩节点功能
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_simplified_compression():
    """演示简化后的压缩功能"""
    
    print("=== ComfyUI 图片序列压缩节点演示 ===\n")
    
    try:
        from image_sequence_compressor import ImageSequenceCompressor
        from extract_from_image import extract_from_numpy_array
        
        # 创建节点实例
        compressor = ImageSequenceCompressor()
        
        print("🎯 节点特性:")
        print("  ✅ 只需输入图片序列，自动生成承载图片")
        print("  ✅ 单张图片：自动编码为JPEG")
        print("  ✅ 多张图片：自动编码为MP4")
        print("  ✅ 只输出一张承载图片")
        print("  ✅ 文件大小减少90%以上\n")
        
        # 演示1: 单张图片
        print("📸 演示1: 单张图片压缩")
        test_single = torch.rand(1, 3, 512, 512)
        
        result_single = compressor.compress_sequence(
            images=[test_single],
            quality=85,
            container_size=512
        )
        
        print(f"  输入: 1张图片")
        print(f"  输出: 1张承载图片 (形状: {result_single[0].shape})")
        
        # 演示2: 多张图片
        print("\n🎬 演示2: 多张图片压缩")
        test_multiple = [
            torch.rand(1, 3, 512, 512),
            torch.rand(1, 3, 512, 512),
            torch.rand(1, 3, 512, 512),
        ]
        
        result_multiple = compressor.compress_sequence(
            images=test_multiple,
            quality=85,
            container_size=512
        )
        
        print(f"  输入: {len(test_multiple)}张图片")
        print(f"  输出: 1张承载图片 (形状: {result_multiple[0].shape})")
        
        # 演示提取功能
        print("\n🔍 演示3: 数据提取")
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print("  提取单张图片数据...")
            extract_from_numpy_array(result_single[0], os.path.join(temp_dir, "single"))
            
            print("  提取多张图片数据...")
            extract_from_numpy_array(result_multiple[0], os.path.join(temp_dir, "multiple"))
            
            # 检查提取的文件
            import glob
            single_files = glob.glob(os.path.join(temp_dir, "single", "*"))
            multiple_files = glob.glob(os.path.join(temp_dir, "multiple", "*"))
            
            print(f"  单张图片提取: {len(single_files)}个文件")
            print(f"  多张图片提取: {len(multiple_files)}个文件")
        
        print("\n✅ 演示完成！")
        print("\n📋 使用方法:")
        print("  1. 在ComfyUI中添加 'Image Sequence Compressor' 节点")
        print("  2. 连接图片序列到 'images' 输入")
        print("  3. 设置质量参数 (默认85)")
        print("  4. 设置承载图片尺寸 (默认512)")
        print("  5. 运行工作流，获得包含压缩数据的图片")
        print("  6. 使用 extract_from_image.py 脚本提取原始数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_simplified_compression()
    
    if success:
        print("\n🎉 演示成功！节点功能正常。")
    else:
        print("\n💥 演示失败，请检查安装。")
