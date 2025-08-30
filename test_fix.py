#!/usr/bin/env python3
"""
测试TT img enc节点的torch兼容性修复
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_torch_compatibility():
    """测试torch兼容性"""
    print("测试torch兼容性...")
    
    try:
        import torch
        print(f"✓ torch版本: {torch.__version__}")
        
        # 测试基本张量操作
        test_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        test_tensor = torch.from_numpy(test_array).float() / 255.0
        
        print(f"✓ 数组转张量成功，形状: {test_tensor.shape}")
        print(f"✓ 张量数据类型: {test_tensor.dtype}")
        print(f"✓ 张量值范围: {test_tensor.min():.3f} - {test_tensor.max():.3f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ torch导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ torch测试失败: {e}")
        return False

def test_node_import():
    """测试节点导入"""
    print("\n测试节点导入...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        print("✓ 节点类导入成功")
        
        # 创建节点实例
        node = TTImgEncNode()
        print("✓ 节点实例创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 节点导入失败: {e}")
        return False

def test_node_output():
    """测试节点输出格式"""
    print("\n测试节点输出格式...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建测试图片
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # 测试单张图片处理
        result = node.process_images([test_image], fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            
            # 检查输出类型
            if hasattr(output, 'cpu'):
                print("✓ 输出是torch张量，具有.cpu()方法")
                print(f"✓ 输出形状: {output.shape}")
                print(f"✓ 输出数据类型: {output.dtype}")
                print(f"✓ 输出值范围: {output.min():.3f} - {output.max():.3f}")
                return True
            else:
                print(f"✗ 输出不是torch张量: {type(output)}")
                return False
        else:
            print("✗ 节点输出为空")
            return False
            
    except Exception as e:
        print(f"✗ 节点输出测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点torch兼容性测试")
    print("=" * 60)
    
    tests = [
        ("torch兼容性", test_torch_compatibility),
        ("节点导入", test_node_import),
        ("节点输出格式", test_node_output),
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
        print("🎉 所有测试通过！节点torch兼容性修复成功")
        print("\n现在可以在ComfyUI中正常使用TT img enc节点了！")
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
