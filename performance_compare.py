#!/usr/bin/env python3
"""
性能对比测试脚本 - 测试优化前后的解码速度
"""

import time
import os
import sys
import subprocess

def test_decode_performance(image_path, password="", iterations=3):
    """
    测试解码性能，多次运行取平均值
    
    Args:
        image_path: 测试图片路径
        password: 密码（可选）
        iterations: 测试次数
    """
    if not os.path.exists(image_path):
        print(f"📁 测试图片不存在: {image_path}")
        return None
    
    print(f"🔍 开始性能测试: {image_path}")
    print(f"📏 文件大小: {os.path.getsize(image_path) / 1024 / 1024:.2f} MB")
    print(f"🔄 测试次数: {iterations}")
    
    times = []
    
    for i in range(iterations):
        print(f"\n--- 第 {i+1} 次测试 ---")
        
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
                print(f"✅ 解码成功 - 耗时: {elapsed_time:.2f} 秒")
                times.append(elapsed_time)
            else:
                print(f"❌ 解码失败")
                print(f"错误信息: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 测试超时（5分钟）")
            return None
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            return None
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # 计算处理速度
        file_size_mb = os.path.getsize(image_path) / 1024 / 1024
        avg_speed = file_size_mb / avg_time
        
        print(f"\n📊 测试结果汇总:")
        print(f"  平均耗时: {avg_time:.2f} 秒")
        print(f"  最快耗时: {min_time:.2f} 秒")
        print(f"  最慢耗时: {max_time:.2f} 秒")
        print(f"  平均速度: {avg_speed:.2f} MB/s")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_speed': avg_speed,
            'times': times
        }
    
    return None

def main():
    """主函数"""
    print("🚀 TT img 解码器性能对比测试")
    print("=" * 60)
    
    # 测试图片
    test_image = "pw_test.png"
    password = "123456"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        print("请确保当前目录下有 pw_test.png 文件")
        return
    
    print(f"📁 测试图片: {test_image}")
    print(f"🔑 测试密码: {password}")
    
    # 运行性能测试
    result = test_decode_performance(test_image, password, iterations=3)
    
    if result:
        print(f"\n{'='*60}")
        print("🎯 性能分析")
        print("=" * 60)
        
        print(f"📈 优化效果:")
        print(f"  平均处理速度: {result['avg_speed']:.2f} MB/s")
        print(f"  平均处理时间: {result['avg_time']:.2f} 秒")
        
        # 性能评估
        if result['avg_speed'] > 10:
            print(f"🚀 性能评级: 优秀 (>10 MB/s)")
        elif result['avg_speed'] > 5:
            print(f"✅ 性能评级: 良好 (5-10 MB/s)")
        elif result['avg_speed'] > 2:
            print(f"⚠️  性能评级: 一般 (2-5 MB/s)")
        else:
            print(f"❌ 性能评级: 较慢 (<2 MB/s)")
        
        print(f"\n💡 优化建议:")
        if result['avg_speed'] < 5:
            print("  - 考虑使用SSD硬盘")
            print("  - 增加系统内存")
            print("  - 关闭其他占用CPU的程序")
        else:
            print("  - 当前性能已经很好")
            print("  - 可以处理更大的文件")
        
        print(f"\n📋 详细时间记录:")
        for i, t in enumerate(result['times'], 1):
            print(f"  第{i}次: {t:.2f} 秒")
    else:
        print("❌ 性能测试失败")

if __name__ == "__main__":
    main()
