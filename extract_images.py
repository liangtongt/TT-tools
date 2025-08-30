#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片序列提取脚本
从压缩的图像文件中提取图片序列
"""

import os
import json
import base64
import zlib
from PIL import Image
import argparse
import sys
import io

def extract_images_from_compressed_file(compressed_file_path, output_directory):
    """从压缩文件中提取图片序列"""
    
    if not os.path.exists(compressed_file_path):
        print(f"错误：文件 {compressed_file_path} 不存在")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # 打开压缩的图像文件
        with Image.open(compressed_file_path) as img:
            # 获取图像尺寸
            width, height = img.size
            
            # 读取底部数据行
            data_length = 0
            for i in range(3):
                if i < width:
                    pixel = img.getpixel((i, height - 1))
                    data_length |= pixel[0] << (16 - i * 8)
            
            if data_length == 0:
                print("错误：无法读取数据长度信息")
                return False
            
            # 计算数据行数
            pixels_per_row = 255
            rows_needed = (data_length + pixels_per_row - 1) // pixels_per_row
            
            # 读取数据
            data_bytes = bytearray()
            for row in range(rows_needed):
                for col in range(pixels_per_row):
                    if col < width and (height - 1 - rows_needed + row) >= 0:
                        pixel = img.getpixel((col, height - 1 - rows_needed + row))
                        data_bytes.append(pixel[0])
            
            # 截取到正确的长度
            data_bytes = data_bytes[:data_length]
            
            # 解析JSON数据
            json_data = data_bytes.decode('utf-8')
            combined_data = json.loads(json_data)
            
            metadata = combined_data["metadata"]
            images_data = combined_data["images"]
            
            print(f"找到 {metadata['image_count']} 张图片")
            print(f"压缩格式: {metadata['format']}")
            print(f"压缩质量: {metadata['quality']}")
            print(f"压缩时间: {metadata['timestamp']}")
            
            # 提取每张图片
            for i, img_data in enumerate(images_data):
                print(f"正在提取图片 {i+1}/{len(images_data)}...")
                
                # 解码和 decompress
                compressed_data = base64.b64decode(img_data)
                original_data = zlib.decompress(compressed_data)
                
                # 创建图像
                img_buffer = io.BytesIO(original_data)
                extracted_img = Image.open(img_buffer)
                
                # 保存图像
                output_filename = f"extracted_image_{i:04d}.png"
                output_path = os.path.join(output_directory, output_filename)
                extracted_img.save(output_path, "PNG")
                
                print(f"  保存到: {output_path}")
                print(f"  尺寸: {extracted_img.size}")
                print(f"  模式: {extracted_img.mode}")
            
            print(f"\n所有图片已成功提取到: {output_directory}")
            return True
            
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="从压缩的图像文件中提取图片序列")
    parser.add_argument("compressed_file", help="压缩的图像文件路径")
    parser.add_argument("output_directory", help="输出目录")
    
    args = parser.parse_args()
    
    success = extract_images_from_compressed_file(args.compressed_file, args.output_directory)
    
    if success:
        print("\n提取完成！")
        sys.exit(0)
    else:
        print("\n提取失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
