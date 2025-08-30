#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度调试TT img节点输出格式
"""

import torch
import numpy as np
import sys
import os
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def deep_debug():
    """深度调试输出格式"""
    
    try:
        from image_sequence_compressor import TTImg
        
        print("=== 🔍 TT img 节点深度调试 ===\n")
        
        # 创建节点实例
        tt_img = TTImg()
        
        # 测试用例：模拟ComfyUI中的实际使用
        print("📸 测试1: 模拟ComfyUI实际使用场景")
        
        # 创建测试图像 - 使用与错误信息中相同的尺寸
        test_image = torch.rand(1, 3, 2048, 2048)
        print(f"  输入图像形状: {test_image.shape}")
        print(f"  输入图像类型: {test_image.dtype}")
        print(f"  输入图像范围: [{test_image.min().item():.3f}, {test_image.max().item():.3f}]")
        
        # 调用节点函数
        print("\n🔄 调用 compress_sequence 函数...")
        result = tt_img.compress_sequence(
            images=[test_image],
            quality=100,
            use_original_size=True
        )
        
        # 详细分析输出
        output_tensor = result[0]
        print(f"\n📊 输出分析:")
        print(f"  输出类型: {type(output_tensor)}")
        print(f"  输出形状: {output_tensor.shape}")
        print(f"  输出数据类型: {output_tensor.dtype}")
        print(f"  输出设备: {output_tensor.device}")
        
        # 检查是否与错误信息匹配
        print(f"\n⚠️  错误格式检查:")
        print(f"  错误信息中的格式: (1, 1, 2048), |u1")
        print(f"  我们的输出格式: {output_tensor.shape}, {output_tensor.dtype}")
        
        if output_tensor.shape == (1, 1, 2048):
            print("  ❌ 输出形状与错误信息完全匹配！")
            print("  🔍 这可能是缓存问题或代码没有正确更新")
        else:
            print("  ✅ 输出形状与错误信息不匹配")
        
        # 转换为numpy进行详细检查
        print(f"\n🔍 转换为numpy检查:")
        numpy_array = output_tensor.cpu().numpy()
        print(f"  numpy形状: {numpy_array.shape}")
        print(f"  numpy类型: {numpy_array.dtype}")
        print(f"  numpy范围: [{numpy_array.min():.3f}, {numpy_array.max():.3f}]")
        
        # 检查每个维度
        if len(numpy_array.shape) == 4:
            batch_size, channels, height, width = numpy_array.shape
            print(f"  批次大小: {batch_size}")
            print(f"  通道数: {channels}")
            print(f"  高度: {height}")
            print(f"  宽度: {width}")
            
            # 检查通道数
            if channels == 1:
                print("  ❌ 通道数错误: 只有1个通道，应该是3个RGB通道")
                print("  🔍 这可能是数据嵌入过程中的问题")
            elif channels == 3:
                print("  ✅ 通道数正确: 3个RGB通道")
            else:
                print(f"  ⚠️  通道数异常: {channels}个通道")
        
        # 检查数据类型
        if numpy_array.dtype == np.uint8:
            print("  ❌ 数据类型错误: uint8，应该是float32")
            print("  🔍 这可能是归一化过程的问题")
        elif numpy_array.dtype == np.float32:
            print("  ✅ 数据类型正确: float32")
        else:
            print(f"  ⚠️  数据类型异常: {numpy_array.dtype}")
        
        # 检查数值范围
        if numpy_array.dtype == np.float32:
            if 0.0 <= numpy_array.min() <= numpy_array.max() <= 1.0:
                print("  ✅ 数值范围正确: [0.0, 1.0]")
            else:
                print(f"  ❌ 数值范围错误: [{numpy_array.min():.3f}, {numpy_array.max():.3f}]")
                print("  🔍 这可能是归一化过程的问题")
        
        # 模拟SaveImage节点的处理
        print(f"\n🖼️  模拟SaveImage节点处理:")
        try:
            # 模拟ComfyUI的SaveImage处理
            if numpy_array.dtype == np.float32:
                # 转换为0-255范围
                img_array = np.clip(numpy_array * 255, 0, 255).astype(np.uint8)
                print(f"  转换后形状: {img_array.shape}")
                print(f"  转换后类型: {img_array.dtype}")
                print(f"  转换后范围: [{img_array.min()}, {img_array.max()}]")
                
                # 尝试创建PIL图像
                from PIL import Image
                if len(img_array.shape) == 4:
                    # 取第一个图像
                    single_img = img_array[0]
                    if single_img.shape[0] == 3:  # CHW格式
                        # 转换为HWC格式
                        single_img = np.transpose(single_img, (1, 2, 0))
                    
                    pil_img = Image.fromarray(single_img)
                    print(f"  ✅ PIL图像创建成功: {pil_img.size}, {pil_img.mode}")
                else:
                    print(f"  ❌ 无法创建PIL图像: 形状不正确 {img_array.shape}")
            else:
                print(f"  ❌ 无法处理非float32数据: {numpy_array.dtype}")
                
        except Exception as e:
            print(f"  ❌ SaveImage模拟失败: {e}")
            traceback.print_exc()
        
        # 检查代码版本
        print(f"\n📝 代码版本检查:")
        try:
            with open("image_sequence_compressor.py", "r", encoding="utf-8") as f:
                content = f.read()
                if "简化并修复输出格式处理" in content:
                    print("  ✅ 代码已更新到最新版本")
                else:
                    print("  ❌ 代码可能不是最新版本")
                    
                if "强制转换为3通道RGB格式" in content:
                    print("  ✅ 包含最新的格式修复代码")
                else:
                    print("  ❌ 缺少最新的格式修复代码")
        except Exception as e:
            print(f"  ❌ 无法读取代码文件: {e}")
        
        print(f"\n🎯 调试总结:")
        if output_tensor.shape == (1, 1, 2048):
            print("  🔴 问题确认: 输出格式与错误信息完全匹配")
            print("  💡 可能原因:")
            print("     1. ComfyUI缓存问题 - 需要重启")
            print("     2. 代码没有正确更新 - 检查文件")
            print("     3. 其他节点影响 - 检查工作流")
        else:
            print("  🟢 输出格式正常，问题可能在其他地方")
        
        return output_tensor.shape != (1, 1, 2048)
        
    except Exception as e:
        print(f"❌ 深度调试失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = deep_debug()
    
    if success:
        print("\n✅ 深度调试完成，输出格式正常")
    else:
        print("\n❌ 深度调试发现问题，需要进一步调查")
