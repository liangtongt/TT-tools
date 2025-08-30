#!/usr/bin/env python3
"""
快速测试TT img enc节点ZIP提取功能
"""

def quick_extraction_test():
    """快速提取测试"""
    print("🔧 快速测试TT img enc节点ZIP提取功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        import numpy as np
        
        node = TTImgEncNode()
        print("✓ 节点导入成功")
        
        # 创建测试数据
        test_content = b"This is a test file for ZIP extraction testing. " * 100
        print(f"✓ 创建测试数据: {len(test_content)} 字节")
        
        # 创建ZIP数据
        zip_data = node._create_zip_with_file(test_content, "txt")
        print(f"✓ 创建ZIP数据: {len(zip_data)} 字节")
        
        # 创建测试图片
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        print("✓ 创建测试图片")
        
        # 嵌入ZIP数据
        embedded_image = node._embed_zip_data_in_image(test_image, zip_data)
        print("✓ ZIP数据嵌入成功")
        
        # 测试提取
        from extract_zip import extract_zip_data_from_image
        
        extracted_data = extract_zip_data_from_image(embedded_image)
        
        if extracted_data is not None:
            print(f"✓ 数据提取成功: {len(extracted_data)} 字节")
            
            # 验证数据完整性
            if extracted_data == zip_data:
                print("🎉 数据完整性验证通过！")
                print(f"   嵌入数据: {len(zip_data)} 字节")
                print(f"   提取数据: {len(extracted_data)} 字节")
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

if __name__ == "__main__":
    success = quick_extraction_test()
    if success:
        print("\n✅ ZIP提取功能测试通过！")
        print("\n现在可以:")
        print("1. 在ComfyUI中使用TT img enc节点")
        print("2. 下载输出的造点图片")
        print("3. 运行: python extract_zip.py <图片路径>")
        print("4. 成功提取隐藏的ZIP文件")
    else:
        print("\n❌ ZIP提取功能测试失败，请检查错误信息")
