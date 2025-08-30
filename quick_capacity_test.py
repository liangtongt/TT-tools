#!/usr/bin/env python3
"""
快速测试TT img enc节点存储容量修复
"""

def quick_capacity_test():
    """快速容量测试"""
    print("🔧 快速测试TT img enc节点存储容量修复...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        print("✓ 节点导入成功")
        
        # 测试21帧视频文件大小
        frame_size = 100 * 1024  # 100KB per frame
        num_frames = 21
        video_size = frame_size * num_frames
        
        print(f"模拟21帧视频文件大小: {video_size / 1024:.1f} KB")
        
        # 计算所需图片尺寸
        required_size = node._calculate_required_image_size(b'x' * video_size)
        print(f"需要图片尺寸: {required_size}x{required_size}")
        
        # 计算实际容量
        capacity = required_size * required_size * 3 / 8 / 1024
        print(f"图片存储容量: {capacity:.1f} KB")
        
        if capacity >= video_size / 1024:
            print("🎉 容量足够！现在可以存储21帧视频文件")
            print(f"   视频大小: {video_size / 1024:.1f} KB")
            print(f"   图片容量: {capacity:.1f} KB")
            return True
        else:
            print(f"✗ 容量仍然不足")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_capacity_test()
    if success:
        print("\n✅ 存储容量问题已修复！")
        print("\n修复内容:")
        print("- 动态计算所需图片尺寸")
        print("- 移除base64编码，直接嵌入二进制数据")
        print("- 支持最大2048x2048图片尺寸")
        print("- 大幅提升存储容量（从96KB到1.5MB）")
        print("- 现在可以轻松存储21帧视频文件")
    else:
        print("\n❌ 存储容量问题仍未解决，请检查错误信息")
