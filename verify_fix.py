#!/usr/bin/env python3
"""
快速验证TT img enc节点修复
"""

def verify_fix():
    """验证修复"""
    print("🔧 验证TT img enc节点修复...")
    
    try:
        # 测试导入
        from tt_img_enc_node import TTImgEncNode
        print("✓ 节点导入成功")
        
        # 创建节点
        node = TTImgEncNode()
        print("✓ 节点创建成功")
        
        # 测试torch张量输入
        import torch
        import numpy as np
        
        # 模拟ComfyUI输入：torch张量，值范围0-1
        test_tensor = torch.rand(1, 100, 100, 3)
        print(f"✓ 创建测试torch张量: {test_tensor.shape}")
        
        # 测试处理
        result = node.process_images([test_tensor])
        if result and len(result) > 0:
            output = result[0]
            print(f"✓ 处理成功，输出形状: {output.shape}")
            
            # 检查输出格式
            if hasattr(output, 'cpu') and len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 修复成功！现在可以处理torch张量输入")
                print(f"   输入: torch张量 {test_tensor.shape}")
                print(f"   输出: torch张量 {output.shape}")
                return True
            else:
                print(f"✗ 输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_fix()
    if success:
        print("\n✅ 节点修复完成！现在可以正常处理ComfyUI的torch张量输入")
        print("\n修复内容:")
        print("- 自动检测输入类型（torch张量或numpy数组）")
        print("- 自动转换torch张量为numpy数组")
        print("- 自动处理值范围（0-1或0-255）")
        print("- 完全兼容ComfyUI的IMAGE类型输入")
    else:
        print("\n❌ 节点仍有问题，请检查错误信息")
