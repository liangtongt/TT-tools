#!/usr/bin/env python3
"""
测试TT img enc节点处理torch张量输入
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_torch_input():
    """测试torch张量输入处理"""
    print("测试torch张量输入处理...")
    
    try:
        import torch
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建torch张量（模拟ComfyUI输入）
        # ComfyUI通常传入的是(batch, height, width, channels)格式，值范围0-1
        test_tensor = torch.rand(1, 256, 256, 3)  # 模拟单张图片
        print(f"✓ 创建测试torch张量，形状: {test_tensor.shape}")
        print(f"✓ 张量值范围: {test_tensor.min():.3f} - {test_tensor.max():.3f}")
        
        # 测试处理
        result = node.process_images([test_tensor], fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            print(f"✓ 处理成功，输出形状: {output.shape}")
            
            # 检查输出格式
            if hasattr(output, 'cpu') and len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 成功！torch张量输入处理正常")
                return True
            else:
                print(f"✗ 输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_multiple_torch_inputs():
    """测试多张torch张量输入"""
    print("\n测试多张torch张量输入...")
    
    try:
        import torch
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建多张torch张量
        test_tensors = []
        for i in range(3):
            tensor = torch.rand(1, 256, 256, 3)
            test_tensors.append(tensor)
        
        print(f"✓ 创建{len(test_tensors)}张测试torch张量")
        
        # 测试处理
        result = node.process_images(test_tensors, fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            print(f"✓ 多张输入处理成功，输出形状: {output.shape}")
            
            # 检查输出格式
            if hasattr(output, 'cpu') and len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 成功！多张torch张量输入处理正常")
                return True
            else:
                print(f"✗ 输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 多张输入处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 多张输入测试失败: {e}")
        return False

def test_mixed_inputs():
    """测试混合输入类型"""
    print("\n测试混合输入类型...")
    
    try:
        import torch
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 混合torch张量和numpy数组
        mixed_inputs = [
            torch.rand(1, 256, 256, 3),  # torch张量
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),  # numpy数组
        ]
        
        print(f"✓ 创建混合输入类型，数量: {len(mixed_inputs)}")
        
        # 测试处理
        result = node.process_images(mixed_inputs, fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            print(f"✓ 混合输入处理成功，输出形状: {output.shape}")
            
            # 检查输出格式
            if hasattr(output, 'cpu') and len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 成功！混合输入类型处理正常")
                return True
            else:
                print(f"✗ 输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 混合输入处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 混合输入测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点torch输入处理测试")
    print("=" * 60)
    
    tests = [
        ("单张torch张量输入", test_torch_input),
        ("多张torch张量输入", test_multiple_torch_inputs),
        ("混合输入类型", test_mixed_inputs),
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
        print("🎉 所有测试通过！torch张量输入处理正常")
        print("\n修复说明:")
        print("- 自动检测输入类型（torch张量或numpy数组）")
        print("- 自动转换torch张量为numpy数组")
        print("- 自动处理值范围（0-1或0-255）")
        print("- 确保与ComfyUI完全兼容")
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
