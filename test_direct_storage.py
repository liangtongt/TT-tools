#!/usr/bin/env python3
"""
测试TT img enc节点直接存储功能
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_direct_storage():
    """测试直接存储功能"""
    print("测试直接存储功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建测试数据（模拟MP4文件）
        test_data = b"fake_mp4_content_for_testing" * 1000  # 约27KB
        print(f"✓ 创建测试数据: {len(test_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(test_data, "mp4")
        print(f"✓ 创建文件头: {len(file_header)} 字节")
        
        # 创建测试图片
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # 嵌入文件数据
        embedded_image = node._embed_file_data_in_image(test_image, file_header)
        print(f"✓ 文件数据嵌入成功")
        
        # 测试提取
        from extract_zip import extract_file_data_from_image
        
        extracted_data, extracted_extension = extract_file_data_from_image(embedded_image)
        
        if extracted_data is not None and extracted_extension is not None:
            print(f"✓ 数据提取成功: {len(extracted_data)} 字节")
            print(f"✓ 扩展名提取成功: {extracted_extension}")
            
            # 验证数据完整性
            if extracted_data == test_data and extracted_extension == "mp4":
                print("🎉 数据完整性验证通过！")
                return True
            else:
                print("✗ 数据完整性验证失败")
                return False
        else:
            print("✗ 数据提取失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点直接存储功能测试")
    print("=" * 60)
    
    tests = [
        ("直接存储测试", test_direct_storage),
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
        print("🎉 所有测试通过！直接存储功能正常")
        print("\n优势:")
        print("- 移除ZIP压缩，减少文件大小")
        print("- 支持更大的原始文件")
        print("- 保持文件扩展名信息")
        print("- 提高存储效率")
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
