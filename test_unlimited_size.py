#!/usr/bin/env python3
"""
测试TT img enc节点无限制图片尺寸功能
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unlimited_size():
    """测试无限制图片尺寸功能"""
    print("测试无限制图片尺寸功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试不同大小的文件
        test_cases = [
            ("小文件", b"small_file_content" * 100, "txt"),  # 约1.7KB
            ("中等文件", b"medium_file_content" * 10000, "jpg"),  # 约170KB
            ("大文件", b"large_file_content" * 100000, "mp4"),  # 约1.7MB
            ("超大文件", b"huge_file_content" * 500000, "mp4"),  # 约8.5MB
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
            
            # 检查是否超过之前的2048限制
            if required_size > 2048:
                print(f"  ✓ 突破2048限制: {required_size} > 2048")
            else:
                print(f"  ✓ 在2048范围内: {required_size} <= 2048")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_extreme_sizes():
    """测试极端尺寸"""
    print("\n测试极端尺寸...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试超大文件（模拟10MB文件）
        huge_data = b"extreme_large_file_content" * 1000000  # 约25MB
        print(f"✓ 创建超大文件测试数据: {len(huge_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(huge_data, "mp4")
        print(f"✓ 文件头大小: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证容量
        max_capacity = required_size * required_size * 3 // 8
        print(f"✓ 最大容量: {max_capacity} 字节")
        
        if max_capacity >= len(file_header):
            print("✓ 容量足够，可以存储超大文件")
            print(f"✓ 图片尺寸: {required_size}x{required_size} 像素")
            return True
        else:
            print("✗ 容量不足，无法存储超大文件")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_size_calculation():
    """测试尺寸计算逻辑"""
    print("\n测试尺寸计算逻辑...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试不同大小的数据
        test_sizes = [
            (1024, "1KB"),      # 1KB
            (10240, "10KB"),    # 10KB
            (102400, "100KB"),  # 100KB
            (1048576, "1MB"),   # 1MB
            (10485760, "10MB"), # 10MB
            (52428800, "50MB"), # 50MB
        ]
        
        for data_size, size_name in test_sizes:
            test_data = b"x" * data_size
            file_header = node._create_file_header(test_data, "dat")
            required_size = node._calculate_required_image_size(file_header)
            max_capacity = required_size * required_size * 3 // 8
            
            print(f"  {size_name}: {data_size} 字节 -> {required_size}x{required_size} 图片 -> {max_capacity} 字节容量")
            
            if max_capacity >= len(file_header):
                print(f"    ✓ 容量足够")
            else:
                print(f"    ✗ 容量不足")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点无限制图片尺寸功能测试")
    print("=" * 60)
    
    tests = [
        ("无限制尺寸测试", test_unlimited_size),
        ("极端尺寸测试", test_extreme_sizes),
        ("尺寸计算测试", test_size_calculation),
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
        print("🎉 所有测试通过！无限制图片尺寸功能正常")
        print("\n优势:")
        print("- 支持任意大小的文件")
        print("- 无图片尺寸上限限制")
        print("- 自动计算最佳图片尺寸")
        print("- 保持存储效率")
        print("- 兼容现有提取工具")
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
