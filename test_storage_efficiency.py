#!/usr/bin/env python3
"""
测试TT img enc节点存储效率优化
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_storage_efficiency():
    """测试存储效率优化"""
    print("测试存储效率优化...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试不同大小的文件
        test_cases = [
            ("小文件", b"small_file_content" * 10, "txt"),      # 约170字节
            ("中等文件", b"medium_file_content" * 1000, "jpg"), # 约17KB
            ("大文件", b"large_file_content" * 10000, "mp4"),   # 约170KB
        ]
        
        for test_name, test_data, extension in test_cases:
            print(f"\n测试 {test_name}: {len(test_data)} 字节")
            
            # 创建文件头
            file_header = node._create_file_header(test_data, extension)
            print(f"  文件头大小: {len(file_header)} 字节")
            
            # 计算所需图片尺寸
            required_size = node._calculate_required_image_size(file_header)
            print(f"  需要图片尺寸: {required_size}x{required_size}")
            
            # 验证尺寸计算
            max_capacity = required_size * required_size * 3 // 8
            if max_capacity >= len(file_header):
                print(f"  ✓ 容量足够: {max_capacity} >= {len(file_header)}")
            else:
                print(f"  ✗ 容量不足: {max_capacity} < {len(file_header)}")
                return False
            
            # 检查是否使用最小尺寸
            if required_size == 64:
                print(f"  ✓ 使用最小尺寸: {required_size}x{required_size}")
            else:
                print(f"  ✓ 动态尺寸: {required_size}x{required_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_minimum_size():
    """测试最小尺寸优化"""
    print("\n测试最小尺寸优化...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试非常小的文件
        tiny_data = b"tiny"  # 4字节
        print(f"✓ 创建小文件测试数据: {len(tiny_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(tiny_data, "txt")
        print(f"✓ 文件头大小: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证是否使用最小尺寸
        if required_size == 64:
            print("✓ 正确使用最小尺寸64x64")
            return True
        else:
            print(f"✗ 未使用最小尺寸: {required_size}")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_size_comparison():
    """测试尺寸对比（优化前后）"""
    print("\n测试尺寸对比...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试不同大小的数据
        test_sizes = [
            (100, "100字节"),
            (1000, "1KB"),
            (10000, "10KB"),
            (100000, "100KB"),
        ]
        
        print("文件大小 -> 优化后尺寸 -> 容量")
        print("-" * 40)
        
        for data_size, size_name in test_sizes:
            test_data = b"x" * data_size
            file_header = node._create_file_header(test_data, "dat")
            required_size = node._calculate_required_image_size(file_header)
            max_capacity = required_size * required_size * 3 // 8
            
            print(f"{size_name:>8} -> {required_size:>4}x{required_size:<4} -> {max_capacity:>6} 字节")
            
            if max_capacity >= len(file_header):
                print(f"    ✓ 容量足够")
            else:
                print(f"    ✗ 容量不足")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_storage_image():
    """测试存储图片创建"""
    print("\n测试存储图片创建...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试创建存储图片
        test_sizes = [64, 128, 256, 512]
        
        for size in test_sizes:
            storage_image = node._create_storage_image(size)
            print(f"✓ 创建 {size}x{size} 存储图片")
            
            # 验证图片属性
            if storage_image.shape == (size, size, 3):
                print(f"  ✓ 尺寸正确: {storage_image.shape}")
            else:
                print(f"  ✗ 尺寸错误: {storage_image.shape}")
                return False
            
            # 验证颜色值（应该是128）
            if np.all(storage_image == 128):
                print(f"  ✓ 颜色值正确: 128")
            else:
                print(f"  ✗ 颜色值错误: {storage_image[0,0]}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点存储效率优化测试")
    print("=" * 60)
    
    tests = [
        ("存储效率测试", test_storage_efficiency),
        ("最小尺寸测试", test_minimum_size),
        ("尺寸对比测试", test_size_comparison),
        ("存储图片测试", test_storage_image),
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
        print("🎉 所有测试通过！存储效率优化成功")
        print("\n优化效果:")
        print("- 移除噪点设置，简化界面")
        print("- 最小图片尺寸从512x512降至64x64")
        print("- 使用纯色背景，提高存储效率")
        print("- 支持更小文件的存储")
        print("- 保持无限制存储能力")
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
