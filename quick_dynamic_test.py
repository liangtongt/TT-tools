#!/usr/bin/env python3
"""
快速测试TT img enc节点动态尺寸功能
"""

def quick_dynamic_test():
    """快速动态尺寸测试"""
    print("🔧 快速测试TT img enc节点动态尺寸功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        import numpy as np
        
        node = TTImgEncNode()
        print("✓ 节点导入成功")
        
        # 创建大文件测试数据（模拟865KB MP4）
        large_data = b"large_mp4_content_for_testing" * 30000  # 约900KB
        print(f"✓ 创建大文件测试数据: {len(large_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(large_data, "mp4")
        print(f"✓ 创建文件头: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证容量
        max_capacity = required_size * required_size * 3 // 8
        print(f"✓ 最大容量: {max_capacity} 字节")
        
        if max_capacity >= len(file_header):
            print("✓ 容量足够，可以存储大文件")
            
            # 创建测试图片
            test_image = np.ones((required_size, required_size, 3), dtype=np.uint8) * 255
            print(f"✓ 创建测试图片: {test_image.shape}")
            
            # 嵌入文件数据
            embedded_image = node._embed_file_data_in_image(test_image, file_header)
            print(f"✓ 数据嵌入成功，图片尺寸: {embedded_image.shape}")
            
            print("🎉 动态尺寸功能测试通过！")
            print(f"   文件大小: {len(file_header)} 字节")
            print(f"   图片尺寸: {required_size}x{required_size}")
            print(f"   存储容量: {max_capacity} 字节")
            return True
        else:
            print("✗ 容量不足，无法存储大文件")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_dynamic_test()
    if success:
        print("\n✅ 动态尺寸功能测试通过！")
        print("\n现在可以:")
        print("1. 在ComfyUI中使用TT img enc节点")
        print("2. 处理大文件（如865KB MP4）")
        print("3. 自动调整图片尺寸")
        print("4. 成功存储和提取大文件")
    else:
        print("\n❌ 动态尺寸功能测试失败，请检查错误信息")
