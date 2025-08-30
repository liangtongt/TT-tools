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
import torch

def extract_images_from_image(image_path, output_directory):
    """从图像中提取压缩的图片序列（简单像素替换）"""
    
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
            
            # 从像素中读取数据长度
            data_length = 0
            for i in range(4):
                x, y = i % width, i // width
                if y < height:
                    pixel = img.getpixel((x, y))
                    # 从像素中提取数据
                    byte_val = pixel[0]  # 使用红色通道
                    data_length = (data_length << 8) | byte_val
            
            if data_length == 0:
                print("错误：无法读取数据长度信息")
                print("提示：这个图像可能没有包含压缩的图片序列数据")
                return False
            
                        # 从像素中读取数据
            data_bytes = bytearray()
            for i in range(data_length):
                pixel_index = i + 4  # 跳过长度信息
                x, y = pixel_index % width, pixel_index // width
                if y < height:
                    pixel = img.getpixel((x, y))
                    # 从像素中提取数据
                    byte_val = pixel[0]  # 使用红色通道
                    data_bytes.append(byte_val)
            
            # 解析JSON数据
            json_data = data_bytes.decode('utf-8')
            combined_data = json.loads(json_data)
            
            metadata = combined_data["metadata"]
            data = combined_data["data"]
            
            print(f"找到 {metadata['image_count']} 张图片")
            print(f"压缩类型: {metadata['type']}")
            print(f"压缩格式: {metadata['format']}")
            print(f"压缩质量: {metadata['quality']}")
            print(f"压缩时间: {metadata['timestamp']}")
            
            # 解码数据
            compressed_data = base64.b64decode(data)
            
            if metadata['type'] == 'single':
                # 单张图片：直接保存JPEG
                output_filename = "extracted_image.jpg"
                output_path = os.path.join(output_directory, output_filename)
                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
                print(f"  保存到: {output_path}")
                
            else:
                # 图片序列：保存MP4
                output_filename = "extracted_sequence.mp4"
                output_path = os.path.join(output_directory, output_filename)
                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
                print(f"  保存到: {output_path}")
                
                # 如果需要提取单帧，可以使用OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(output_path)
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 保存单帧
                        frame_filename = f"extracted_frame_{frame_count:04d}.jpg"
                        frame_path = os.path.join(output_directory, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_count += 1
                    
                    cap.release()
                    print(f"  提取了 {frame_count} 帧到单独文件")
                    
                except ImportError:
                    print("  注意：需要安装opencv-python来提取单帧")
            
            print(f"\n所有文件已成功提取到: {output_directory}")
            return True
            
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        return False

def extract_from_numpy_array(image_array, output_directory):
    """从numpy数组格式的图像中提取压缩的图片序列（简单像素替换）"""
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # 处理Tensor对象，转换为numpy数组
        if torch.is_tensor(image_array):
            img_array = image_array.cpu().numpy()
            # 处理批次维度
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            # Tensor通常是CHW格式，需要转换为HWC格式
            if len(img_array.shape) == 3 and img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
        else:
            img_array = image_array
            # 处理numpy数组的批次维度
            if len(img_array.shape) == 4:
                img_array = img_array[0]
        
        # 转换numpy数组为PIL图像
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                # 0-1范围，转换为0-255
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        else:
            img_array = img_array
        
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
        
        # 从像素中读取数据长度
        data_length = 0
        for i in range(4):
            x, y = i % width, i // width
            if y < height:
                pixel = img.getpixel((x, y))
                # 从像素中提取数据
                byte_val = pixel[0]  # 使用红色通道
                data_length = (data_length << 8) | byte_val
        
        if data_length == 0:
            print("错误：无法读取数据长度信息")
            print("提示：这个图像可能没有包含压缩的图片序列数据")
            return False
        
        # 从像素中读取数据
        data_bytes = bytearray()
        for i in range(data_length):
            pixel_index = i + 4  # 跳过长度信息
            x, y = pixel_index % width, pixel_index // width
            if y < height:
                pixel = img.getpixel((x, y))
                # 从像素中提取数据
                byte_val = pixel[0]  # 使用红色通道
                data_bytes.append(byte_val)
        
        # 解析JSON数据
        json_data = data_bytes.decode('utf-8')
        combined_data = json.loads(json_data)
        
        metadata = combined_data["metadata"]
        data = combined_data["data"]
        
        print(f"找到 {metadata['image_count']} 张图片")
        print(f"压缩类型: {metadata['type']}")
        print(f"压缩格式: {metadata['format']}")
        print(f"压缩质量: {metadata['quality']}")
        print(f"压缩时间: {metadata['timestamp']}")
        
        # 解码数据
        compressed_data = base64.b64decode(data)
        
        if metadata['type'] == 'single':
            # 单张图片：直接保存JPEG
            output_filename = "extracted_image.jpg"
            output_path = os.path.join(output_directory, output_filename)
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
            print(f"  保存到: {output_path}")
            
        else:
            # 图片序列：保存MP4
            output_filename = "extracted_sequence.mp4"
            output_path = os.path.join(output_directory, output_filename)
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
            print(f"  保存到: {output_path}")
            
            # 如果需要提取单帧，可以使用OpenCV
            try:
                import cv2
                cap = cv2.VideoCapture(output_path)
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 保存单帧
                    frame_filename = f"extracted_frame_{frame_count:04d}.jpg"
                    frame_path = os.path.join(output_directory, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                
                cap.release()
                print(f"  提取了 {frame_count} 帧到单独文件")
                
            except ImportError:
                print("  注意：需要安装opencv-python来提取单帧")
        
        print(f"\n所有文件已成功提取到: {output_directory}")
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
