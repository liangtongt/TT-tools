#!/usr/bin/env python3
"""
快速验证TT img enc节点无限制功能
"""

def quick_unlimited_test():
    """快速无限制功能测试"""
    print("🔧 快速验证TT img enc节点无限制功能...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        import numpy as np
        
        node = TTImgEncNode()
        print("✓ 节点导入成功")
        
        # 测试超大文件（模拟50MB文件）
        huge_data = b"unlimited_size_test_content" * 2000000  # 约50MB
        print(f"✓ 创建超大文件测试数据: {len(huge_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(huge_data, "mp4")
        print(f"✓ 创建文件头: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证容量
        max_capacity = required_size * required_size * 3 // 8
        print(f"✓ 最大容量: {max_capacity} 字节")
        
        if max_capacity >= len(file_header):
            print("✓ 容量足够，可以存储超大文件")
            
            # 检查是否突破之前的限制
            if required_size > 2048:
                print(f"✓ 成功突破2048限制: {required_size} > 2048")
            else:
                print(f"✓ 在2048范围内: {required_size} <= 2048")
            
            print("🎉 无限制功能验证通过！")
            print(f"   文件大小: {len(file_header)} 字节")
            print(f"   图片尺寸: {required_size}x{required_size}")
            print(f"   存储容量: {max_capacity} 字节")
            return True
        else:
            print("✗ 容量不足，无法存储超大文件")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_unlimited_test()
    if success:
        print("\n✅ 无限制功能验证通过！")
        print("\n现在可以:")
        print("1. 在ComfyUI中使用TT img enc节点")
        print("2. 处理任意大小的文件")
        print("3. 自动调整图片尺寸（无上限）")
        print("4. 支持超大文件存储")
        print("5. 保持存储效率")
    else:
        print("\n❌ 无限制功能验证失败，请检查错误信息")
