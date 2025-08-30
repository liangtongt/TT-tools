#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试图片序列的脚本
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_images(output_dir="test_images", count=5):
    """创建测试图片序列"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建不同颜色和内容的测试图片
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
        (128, 128, 128), # 灰色
        (255, 128, 0),  # 橙色
    ]
    
    for i in range(count):
        # 创建512x512的图片
        width, height = 512, 512
        
        # 随机选择背景色
        bg_color = random.choice(colors)
        
        # 创建图片
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 添加文字
        try:
            # 尝试使用默认字体
            font = ImageFont.load_default()
        except:
            # 如果没有默认字体，使用系统字体
            font = ImageFont.truetype("arial.ttf", 60)
        
        # 添加图片编号
        text = f"Test Image {i+1}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算文字位置（居中）
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # 绘制文字（白色，带黑色边框）
        draw.text((x+2, y+2), text, fill=(0, 0, 0), font=font)  # 黑色边框
        draw.text((x, y), text, fill=(255, 255, 255), font=font)  # 白色文字
        
        # 添加一些几何图形
        if i % 2 == 0:
            # 绘制圆形
            circle_center = (width // 4, height // 4)
            circle_radius = 50
            draw.ellipse([
                circle_center[0] - circle_radius,
                circle_center[1] - circle_radius,
                circle_center[0] + circle_radius,
                circle_center[1] + circle_radius
            ], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        else:
            # 绘制矩形
            rect_x = width // 4
            rect_y = height // 4
            rect_size = 100
            draw.rectangle([
                rect_x, rect_y,
                rect_x + rect_size, rect_y + rect_size
            ], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        
        # 保存图片
        filename = f"test_image_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG")
        
        print(f"创建测试图片: {filepath}")
    
    print(f"\n成功创建 {count} 张测试图片到 {output_dir} 目录")

if __name__ == "__main__":
    create_test_images()
