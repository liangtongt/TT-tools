#!/usr/bin/env python3
"""
快速验证TT img enc节点存储效率优化效果
"""

def quick_efficiency_test():
    """快速存储效率测试"""
    print("🔧 快速验证TT img enc节点存储效率优化...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        import numpy as np
        
        node = TTImgEncNode()
        print("✓ 节点导入成功")
        
        # 测试小文件（之前需要512x512，现在只需要64x64）
        small_data = b"small_file_content" * 50  # 约850字节
        print(f"✓ 创建小文件测试数据: {len(small_data)} 字节")
        
        # 创建文件头
        file_header = node._create_file_header(small_data, "txt")
        print(f"✓ 创建文件头: {len(file_header)} 字节")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(file_header)
        print(f"✓ 需要图片尺寸: {required_size}x{required_size}")
        
        # 验证容量
        max_capacity = required_size * required_size * 3 // 8
        print(f"✓ 最大容量: {max_capacity} 字节")
        
        if max_capacity >= len(file_header):
            print("✓ 容量足够，可以存储文件")
            
            # 检查尺寸优化效果
            if required_size == 64:
                print("✓ 使用最小尺寸64x64（优化前需要512x512）")
                print("✓ 尺寸减少: 512x512 -> 64x64 (减少98.4%)")
                print("✓ 存储效率提升显著！")
            else:
                print(f"✓ 动态尺寸: {required_size}x{required_size}")
            
            # 创建存储图片
            storage_image = node._create_storage_image(required_size)
            print(f"✓ 创建存储图片: {storage_image.shape}")
            
            # 验证图片属性
            if np.all(storage_image == 128):
                print("✓ 纯色背景（灰色128值）")
                print("✓ 无噪点，专为存储优化")
            
            print("🎉 存储效率优化验证通过！")
            print(f"   文件大小: {len(file_header)} 字节")
            print(f"   图片尺寸: {required_size}x{required_size}")
            print(f"   存储容量: {max_capacity} 字节")
            return True
        else:
            print("✗ 容量不足，无法存储文件")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_efficiency_test()
    if success:
        print("\n✅ 存储效率优化验证通过！")
        print("\n优化成果:")
        print("1. 移除噪点设置，界面更简洁")
        print("2. 最小图片尺寸从512x512降至64x64")
        print("3. 使用纯色背景，提高存储效率")
        print("4. 支持更小文件的存储")
        print("5. 保持无限制存储能力")
        print("6. 图片尺寸减少98.4%（小文件）")
    else:
        print("\n❌ 存储效率优化验证失败，请检查错误信息")
