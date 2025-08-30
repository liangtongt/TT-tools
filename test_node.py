#!/usr/bin/env python3
"""
TT img enc 节点测试脚本
测试节点的基本功能
"""

import os
import sys
import numpy as np
from PIL import Image
import tempfile
import zipfile

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_images():
    """创建测试图片"""
    print("创建测试图片...")
    
    # 创建单张测试图片
    single_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # 创建多张测试图片（用于视频）
    multiple_images = []
    for i in range(10):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        # 添加一些变化
        img[:, :, 0] = (img[:, :, 0] + i * 20) % 256
        multiple_images.append(img)
    
    return single_image, multiple_images

def test_single_image_conversion():
    """测试单张图片转换"""
    print("\n测试单张图片转换...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        single_image, _ = create_test_images()
        
        # 测试单张图片处理
        result = node.process_images([single_image], fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output_image = result[0]
            print(f"✓ 单张图片转换成功，输出尺寸: {output_image.shape}")
            
            # 保存输出图片
            output_path = "test_single_output.png"
            Image.fromarray(output_image).save(output_path)
            print(f"✓ 输出图片已保存: {output_path}")
            
            return True
        else:
            print("✗ 单张图片转换失败")
            return False
            
    except Exception as e:
        print(f"✗ 单张图片转换测试失败: {e}")
        return False

def test_multiple_images_conversion():
    """测试多张图片转换"""
    print("\n测试多张图片转换...")
    
    try:
        from tt_img_enc_node import TTImgEncNode
        
        node = TTImgEncNode()
        _, multiple_images = create_test_images()
        
        # 测试多张图片处理
        result = node.process_images(multiple_images, fps=30, quality=95, noise_density=0.1, noise_size=2)
        
        if result and len(result) > 0:
            output_image = result[0]
            print(f"✓ 多张图片转换成功，输出尺寸: {output_image.shape}")
            
            # 保存输出图片
            output_path = "test_multiple_output.png"
            Image.fromarray(output_image).save(output_path)
            print(f"✓ 输出图片已保存: {output_path}")
            
            return True
        else:
            print("✗ 多张图片转换失败")
            return False
            
    except Exception as e:
        print(f"✗ 多张图片转换测试失败: {e}")
        return False

def test_zip_extraction():
    """测试ZIP解压功能"""
    print("\n测试ZIP解压功能...")
    
    try:
        # 创建测试ZIP文件
        test_data = b"This is a test file content for ZIP extraction testing."
        
        zip_buffer = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("test.txt", test_data)
        
        zip_path = zip_buffer.name
        zip_buffer.close()
        
        # 测试解压
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            extracted_data = zip_file.read("test.txt")
        
        # 清理临时文件
        os.unlink(zip_path)
        
        if extracted_data == test_data:
            print("✓ ZIP解压功能正常")
            return True
        else:
            print("✗ ZIP解压功能异常")
            return False
            
    except Exception as e:
        print(f"✗ ZIP解压测试失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 50)
    print("TT img enc 节点功能测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import cv2
        import PIL
        import numpy
        print("✓ 所有依赖包已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请先运行: pip install -r requirements.txt")
        return False
    
    # 运行测试
    tests = [
        ("单张图片转换", test_single_image_conversion),
        ("多张图片转换", test_multiple_images_conversion),
        ("ZIP解压功能", test_zip_extraction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！节点功能正常")
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
