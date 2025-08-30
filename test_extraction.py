#!/usr/bin/env python3
"""
测试TT img enc节点ZIP提取功能
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_zip_creation_and_extraction():
    """测试ZIP创建和提取"""
    print("测试ZIP创建和提取...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建测试数据
        test_data = b"This is a test file content for ZIP extraction testing."
        print(f"测试数据: {len(test_data)} 字节")
        
        # 创建ZIP数据
        zip_data = node._create_zip_with_file(test_data, "txt")
        print(f"ZIP数据: {len(zip_data)} 字节")
        
        # 创建测试图片
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # 嵌入ZIP数据
        embedded_image = node._embed_zip_data_in_image(test_image, zip_data)
        print(f"嵌入后图片尺寸: {embedded_image.shape}")
        
        # 测试提取
        from extract_zip import extract_zip_data_from_image
        
        extracted_data = extract_zip_data_from_image(embedded_image)
        
        if extracted_data is not None:
            print(f"✓ 提取成功: {len(extracted_data)} 字节")
            
            # 验证数据完整性
            if extracted_data == zip_data:
                print("✓ 数据完整性验证通过")
                return True
            else:
                print("✗ 数据完整性验证失败")
                return False
        else:
            print("✗ 提取失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_large_file():
    """测试大文件处理"""
    print("\n测试大文件处理...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建较大的测试数据（接近512x512图片的容量限制）
        # 512x512x3/8 = 98304 字节 ≈ 96KB
        test_data = b"x" * 90000  # 90KB
        
        print(f"测试数据: {len(test_data)} 字节")
        
        # 创建ZIP数据
        zip_data = node._create_zip_with_file(test_data, "dat")
        print(f"ZIP数据: {len(zip_data)} 字节")
        
        # 测试容量检查
        try:
            # 创建测试图片
            test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            
            # 嵌入ZIP数据
            embedded_image = node._embed_zip_data_in_image(test_image, zip_data)
            print(f"✓ 大文件嵌入成功")
            
            # 测试提取
            from extract_zip import extract_zip_data_from_image
            
            extracted_data = extract_zip_data_from_image(embedded_image)
            
            if extracted_data is not None and extracted_data == zip_data:
                print("✓ 大文件提取成功")
                return True
            else:
                print("✗ 大文件提取失败")
                return False
                
        except ValueError as e:
            if "太大" in str(e):
                print(f"✓ 容量检查正常工作: {e}")
                return True
            else:
                print(f"✗ 意外的容量检查错误: {e}")
                return False
                
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_image_save_and_load():
    """测试图片保存和加载"""
    print("\n测试图片保存和加载...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        from PIL import Image
        
        node = TTImgEncNode()
        
        # 创建测试数据
        test_data = b"Test content for image save/load testing."
        zip_data = node._create_zip_with_file(test_data, "txt")
        
        # 创建并嵌入数据
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        embedded_image = node._embed_zip_data_in_image(test_image, zip_data)
        
        # 保存图片
        temp_image_path = "test_embedded_image.png"
        pil_image = Image.fromarray(embedded_image)
        pil_image.save(temp_image_path)
        print(f"✓ 图片保存成功: {temp_image_path}")
        
        # 重新加载图片
        loaded_image = Image.open(temp_image_path)
        loaded_array = np.array(loaded_image)
        print(f"✓ 图片加载成功: {loaded_array.shape}")
        
        # 测试提取
        from extract_zip import extract_zip_data_from_image
        
        extracted_data = extract_zip_data_from_image(loaded_array)
        
        if extracted_data is not None and extracted_data == zip_data:
            print("✓ 保存/加载后提取成功")
            
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print("✓ 临时文件清理完成")
            
            return True
        else:
            print("✗ 保存/加载后提取失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点ZIP提取功能测试")
    print("=" * 60)
    
    tests = [
        ("ZIP创建和提取测试", test_zip_creation_and_extraction),
        ("大文件处理测试", test_large_file),
        ("图片保存和加载测试", test_image_save_and_load),
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
        print("🎉 所有测试通过！ZIP提取功能正常")
        print("\n使用方法:")
        print("1. 在ComfyUI中使用TT img enc节点")
        print("2. 下载输出的造点图片")
        print("3. 运行: python extract_zip.py <图片路径>")
        print("4. 获得隐藏的ZIP文件")
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
