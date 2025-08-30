#!/usr/bin/env python3
"""
快速测试TT img enc节点修复
"""

def quick_test():
    """快速测试"""
    print("🔧 快速测试TT img enc节点修复...")
    
    try:
        # 测试导入
        from tt_img_enc_node import TTImgEncNode
        print("✓ 节点导入成功")
        
        # 创建节点
        node = TTImgEncNode()
        print("✓ 节点创建成功")
        
        # 测试输出格式
        import numpy as np
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = node.process_images([test_img])
        if result and len(result) > 0:
            output = result[0]
            print(f"✓ 输出形状: {output.shape}")
            
            # 检查batch维度
            if len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 修复成功！现在只输出一张图片")
                print(f"   输出格式: (1, {output.shape[1]}, {output.shape[2]}, {output.shape[3]})")
                return True
            else:
                print(f"✗ 输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 输出为空")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ 节点修复完成，可以在ComfyUI中正常使用！")
    else:
        print("\n❌ 节点仍有问题，请检查错误信息")
