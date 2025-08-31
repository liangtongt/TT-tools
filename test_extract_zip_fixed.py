#!/usr/bin/env python3
"""
测试修改后的extract_zip.py的RGBA兼容性
"""

import os
import subprocess
import sys

def test_image_extraction(image_path):
    """测试单个图片的提取功能"""
    print(f"\n=== 测试图片: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return False
    
    try:
        # 运行extract_zip.py
        result = subprocess.run([
            sys.executable, 'extract_zip.py', image_path
        ], capture_output=True, text=True, encoding='utf-8')
        
        # 输出结果
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误:")
            print(result.stderr)
        
        # 检查是否成功
        if result.returncode == 0 and "隐藏文件提取成功" in result.stdout:
            print("✅ 提取成功")
            return True
        else:
            print("❌ 提取失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 测试修改后的extract_zip.py RGBA兼容性")
    
    # 测试图片列表
    test_images = [
        'test3.png',  # RGB格式，包含有效数据
        'test4.png',  # RGBA格式，测试转换功能
        'test.png',   # RGB格式
        'test1.png'   # RGB格式
    ]
    
    success_count = 0
    total_count = len(test_images)
    
    for image in test_images:
        if test_image_extraction(image):
            success_count += 1
    
    print(f"\n📊 测试结果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有测试通过！extract_zip.py RGBA兼容性修复成功！")
    else:
        print("⚠️ 部分测试失败，请检查问题")

if __name__ == "__main__":
    main()
