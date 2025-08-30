#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试图像
"""

from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    """创建测试图像"""
    
    # 创建512x512的测试图像
    width, height = 512, 512
    
    # 创建渐变背景
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 添加渐变
    for y in range(height):
        intensity = int(255 - (y / height) * 100)
        color = (intensity, intensity, intensity)
        draw.line([(0, y), (width, y)], fill=color)
    
    # 添加一些图案
    for i in range(10):
        x = (i * 50) % width
        y = (i * 30) % height
        size = 20
        color = (100 + i * 15, 150 + i * 10, 200 + i * 5)
        draw.ellipse([x, y, x + size, y + size], fill=color)
    
    # 保存图像
    filename = "test_image.png"
    img.save(filename, "PNG")
    print(f"✅ 测试图像已创建: {filename}")
    print(f"  尺寸: {width}x{height}")
    print(f"  格式: PNG")
    
    return filename

if __name__ == "__main__":
    create_test_image()
