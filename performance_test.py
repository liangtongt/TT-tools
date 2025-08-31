#!/usr/bin/env python3
"""
性能测试脚本 - 测试解码速度优化效果
"""

import time
import os
import sys
import subprocess

def test_decode_performance(image_path, password=""):
    """
    测试解码性能
    
    Args:
        image_path: 测试图片路径
        password: 密码（可选）
    """
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return
    
    print(f"🔍 开始性能测试: {image_path}")
    print(f"📏 文件大小: {os.path.getsize(image_path) / 1024 / 1024:.2f} MB")
    
    # 构建命令
    cmd = ["python", "tt_img_dec_pw_loc.py", image_path]
    if password:
        cmd.append(password)
    
    # 执行测试
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 解码成功")
            print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
            
            # 计算处理速度
            file_size_mb = os.path.getsize(image_path) / 1024 / 1024
            speed = file_size_mb / elapsed_time
            print(f"🚀 处理速度: {speed:.2f} MB/s")
            
            return elapsed_time, speed
        else:
            print(f"❌ 解码失败")
            print(f"错误信息: {result.stderr}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时（5分钟）")
        return None, None
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return None, None

def main():
    """主函数"""
    print("🚀 TT img 解码器性能测试")
    print("=" * 50)
    
    # 查找测试图片
    test_images = []
    
    # 查找可能的测试图片
    for file in os.listdir("."):
        if file.endswith((".png", ".jpg", ".jpeg")) and "test" in file.lower():
            test_images.append(file)
    
    if not test_images:
        print("❌ 未找到测试图片")
        print("请确保当前目录下有测试图片文件")
        return
    
    print(f"📁 找到 {len(test_images)} 个测试图片:")
    for i, img in enumerate(test_images, 1):
        print(f"  {i}. {img}")
    
    # 测试每个图片
    results = []
    
    for img in test_images:
        print(f"\n{'='*20} 测试 {img} {'='*20}")
        
        # 尝试无密码解码
        time_taken, speed = test_decode_performance(img)
        
        if time_taken is not None:
            results.append({
                'image': img,
                'time': time_taken,
                'speed': speed,
                'password': False
            })
        else:
            # 如果无密码失败，尝试有密码
            print("🔄 尝试密码保护模式...")
            time_taken, speed = test_decode_performance(img, "test_password")
            
            if time_taken is not None:
                results.append({
                    'image': img,
                    'time': time_taken,
                    'speed': speed,
                    'password': True
                })
    
    # 输出测试结果
    if results:
        print(f"\n{'='*50}")
        print("📊 性能测试结果汇总")
        print("=" * 50)
        
        total_time = sum(r['time'] for r in results)
        avg_speed = sum(r['speed'] for r in results) / len(results)
        
        print(f"📈 总测试图片数: {len(results)}")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        print(f"🚀 平均处理速度: {avg_speed:.2f} MB/s")
        
        print(f"\n📋 详细结果:")
        for result in results:
            password_info = "（密码保护）" if result['password'] else "（无密码）"
            print(f"  {result['image']} {password_info}")
            print(f"    耗时: {result['time']:.2f} 秒")
            print(f"    速度: {result['speed']:.2f} MB/s")
    else:
        print("❌ 所有测试都失败了")

if __name__ == "__main__":
    main()
