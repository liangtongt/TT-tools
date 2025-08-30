#!/usr/bin/env python3
"""
测试TT img enc节点确保只输出一张图片
"""

import os
import sys
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_single_output():
    """测试节点只输出一张图片"""
    print("测试节点输出格式...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建测试图片
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # 测试单张图片处理
        result = node.process_images([test_image], fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            
            print(f"✓ 输出类型: {type(output)}")
            print(f"✓ 输出形状: {output.shape}")
            
            # 检查是否是torch张量
            if hasattr(output, 'cpu'):
                print("✓ 输出是torch张量")
                
                # 检查batch维度
                if len(output.shape) == 4:
                    batch_size = output.shape[0]
                    print(f"✓ Batch大小: {batch_size}")
                    
                    if batch_size == 1:
                        print("🎉 成功！节点只输出一张图片")
                        return True
                    else:
                        print(f"✗ 错误！节点输出了{batch_size}张图片")
                        return False
                else:
                    print(f"✗ 错误！输出形状不正确: {output.shape}")
                    return False
            else:
                print(f"✗ 输出不是torch张量: {type(output)}")
                return False
        else:
            print("✗ 节点输出为空")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_multiple_inputs():
    """测试多张输入图片的处理"""
    print("\n测试多张输入图片处理...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        
        # 创建多张测试图片
        test_images = []
        for i in range(5):
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            test_images.append(img)
        
        # 测试多张图片处理
        result = node.process_images(test_images, fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output = result[0]
            
            print(f"✓ 多张输入处理成功")
            print(f"✓ 输出形状: {output.shape}")
            
            # 检查batch维度
            if len(output.shape) == 4 and output.shape[0] == 1:
                print("🎉 多张输入也正确输出一张图片")
                return True
            else:
                print(f"✗ 多张输入输出格式错误: {output.shape}")
                return False
        else:
            print("✗ 多张输入处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 多张输入测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("TT img enc 节点单张输出测试")
    print("=" * 60)
    
    tests = [
        ("单张输出测试", test_single_output),
        ("多张输入测试", test_multiple_inputs),
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
        print("🎉 所有测试通过！节点现在只输出一张图片")
        print("\n修复说明:")
        print("- 添加了batch维度 (unsqueeze(0))")
        print("- 输出格式: (1, height, width, channels)")
        print("- 确保ComfyUI只显示一张图片")
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
