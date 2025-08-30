#!/usr/bin/env python3
"""
测试TT img enc节点动态图片尺寸功能
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dynamic_size():
    """测试动态图片尺寸功能"""
    print("测试动态图片尺寸功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 测试不同大小的文件
        test_cases = [
            ("小文件", b"small_file_content" * 100, "txt"),  # 约1.7KB
            ("中等文件", b"medium_file_content" * 1000, "jpg"),  # 约17KB
            ("大文件", b"large_file_content" * 10000, "mp4"),  # 约170KB
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
            
            # 创建测试图片
            test_image = np.ones((required_size, required_size, 3), dtype=np.uint8) * 255
            
            # 嵌入文件数据
            embedded_image = node._embed_file_data_in_image(test_image, file_header)
            print(f"  ✓ 数据嵌入成功，图片尺寸: {embedded_image.shape}")
            
            # 测试提取
            from extract_zip import extract_file_data_from_image
            
            extracted_data, extracted_extension = extract_file_data_from_image(embedded_image)
            
            if extracted_data is not None and extracted_extension is not None:
                if extracted_data == test_data and extracted_extension == extension:
                    print(f"  ✓ 数据完整性验证通过")
                else:
                    print(f"  ✗ 数据完整性验证失败")
                    return False
            else:
                print(f"  ✗ 数据提取失败")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_large_file_capacity():
    """测试大文件容量"""
    print("\n测试大文件容量...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 模拟865KB的MP4文件
        large_data = b"large_mp4_content" * 50000  # 约800KB
        print(f"✓ 创建大文件测试数据: {len(large_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(large_data, "mp4")
        print(f"✓ 文件头大小: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证容量
        max_capacity = required_size * required_size * 3 // 8
        print(f"✓ 最大容量: {max_capacity} 字节")
        
        if max_capacity >= len(file_header):
            print("✓ 容量足够，可以存储大文件")
            return True
        else:
            print("✗ 容量不足，无法存储大文件")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点动态图片尺寸功能测试")
    print("=" * 60)
    
    tests = [
        ("动态尺寸测试", test_dynamic_size),
        ("大文件容量测试", test_large_file_capacity),
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
        print("🎉 所有测试通过！动态图片尺寸功能正常")
        print("\n优势:")
        print("- 支持大文件存储（如865KB MP4）")
        print("- 自动调整图片尺寸")
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
