#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从图像中提取压缩的图片序列
支持从ComfyUI输出的图像中提取隐藏的图片序列
"""

import os
import json
import base64
import zlib
from PIL import Image
import argparse
import sys
import io
import numpy as np

def extract_images_from_image(image_path, output_directory):
    """从图像中提取压缩的图片序列"""
    
    if not os.path.exists(image_path):
        print(f"错误：图像文件 {image_path} 不存在")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # 打开图像文件
        with Image.open(image_path) as img:
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
                print("提示：这个图像可能没有包含压缩的图片序列数据")
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

def extract_from_numpy_array(image_array, output_directory):
    """从numpy数组格式的图像中提取压缩的图片序列"""
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # 转换numpy数组为PIL图像
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                # 0-1范围，转换为0-255
                img_array = (image_array * 255).astype(np.uint8)
            else:
                img_array = image_array.astype(np.uint8)
        else:
            img_array = image_array
        
        # 确保图像是3通道RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = Image.fromarray(img_array, 'RGB')
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img = Image.fromarray(img_array, 'RGBA')
            img = img.convert('RGB')
        else:
            img = Image.fromarray(img_array, 'RGB')
        
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
            print("提示：这个图像可能没有包含压缩的图片序列数据")
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
    parser = argparse.ArgumentParser(description="从图像中提取压缩的图片序列")
    parser.add_argument("image_file", help="包含压缩数据的图像文件路径")
    parser.add_argument("output_directory", help="输出目录")
    
    args = parser.parse_args()
    
    success = extract_images_from_image(args.image_file, args.output_directory)
    
    if success:
        print("\n提取完成！")
        sys.exit(0)
    else:
        print("\n提取失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
