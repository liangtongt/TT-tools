#!/usr/bin/env python3
"""
测试TT img enc节点存储容量修复
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_storage_capacity():
    """测试存储容量"""
    print("测试存储容量...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建不同大小的测试数据
        test_sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
        
        for size in test_sizes:
            print(f"\n测试 {size} 字节数据...")
            
            # 创建测试数据
            test_data = np.random.bytes(size)
            
            # 测试容量计算
            required_size = node._calculate_required_image_size(test_data)
            print(f"  数据大小: {size} 字节")
            print(f"  需要图片尺寸: {required_size}x{required_size}")
            print(f"  图片容量: {required_size * required_size * 3 / 8:.0f} 字节")
            
            # 验证容量是否足够
            if required_size * required_size * 3 / 8 >= size:
                print(f"  ✓ 容量足够")
            else:
                print(f"  ✗ 容量不足")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_large_video_simulation():
    """测试大视频文件模拟"""
    print("\n测试大视频文件模拟...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 模拟21帧视频文件（假设每帧100KB）
        frame_size = 100 * 1024  # 100KB per frame
        num_frames = 21
        total_size = frame_size * num_frames
        
        print(f"模拟 {num_frames} 帧视频，总大小: {total_size / 1024:.1f} KB")
        
        # 创建测试数据
        test_data = np.random.bytes(total_size)
        
        # 测试容量计算
        required_size = node._calculate_required_image_size(test_data)
        print(f"需要图片尺寸: {required_size}x{required_size}")
        print(f"图片容量: {required_size * required_size * 3 / 8 / 1024:.1f} KB")
        
        # 验证容量是否足够
        if required_size * required_size * 3 / 8 >= total_size:
            print("✓ 容量足够存储21帧视频")
            return True
        else:
            print("✗ 容量不足")
            return False
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_embedding_process():
    """测试嵌入过程"""
    print("\n测试嵌入过程...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建中等大小的测试数据
        test_data = np.random.bytes(50 * 1024)  # 50KB
        
        print(f"测试数据大小: {len(test_data)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(test_data)
        print(f"需要图片尺寸: {required_size}x{required_size}")
        
        # 创建测试图片
        test_image = np.ones((required_size, required_size, 3), dtype=np.uint8) * 255
        
        # 测试嵌入
        embedded_image = node._embed_binary_data_in_image(test_image, test_data)
        
        print(f"✓ 嵌入成功，输出图片尺寸: {embedded_image.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点存储容量测试")
    print("=" * 60)
    
    tests = [
        ("存储容量测试", test_storage_capacity),
        ("大视频文件模拟", test_large_video_simulation),
        ("嵌入过程测试", test_embedding_process),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！存储容量问题已修复")
        print("\n修复说明:")
        print("- 动态计算所需图片尺寸")
        print("- 移除base64编码，直接嵌入二进制数据")
        print("- 支持最大2048x2048图片尺寸")
        print("- 大幅提升存储容量")
        return True
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)
