#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Image Sequence Compressor 演示脚本
展示如何使用图片序列压缩功能
"""

import os
import sys
import numpy as np
from PIL import Image
import json
import base64
import zlib
import io

# 添加当前目录到Python路径，以便导入节点
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_images():
    """创建演示用的图片序列"""
    print("创建演示图片...")
    
    # 创建3张简单的测试图片
    images = []
    for i in range(3):
        # 创建256x256的图片
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # 设置不同的背景色
        if i == 0:
            img_array[:, :] = [255, 0, 0]  # 红色
        elif i == 1:
            img_array[:, :] = [0, 255, 0]  # 绿色
        else:
            img_array[:, :] = [0, 0, 255]  # 蓝色
        
        # 在图片中心添加白色文字
        img = Image.fromarray(img_array)
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.truetype("arial.ttf", 40)
        
        text = f"Demo {i+1}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (256 - text_width) // 2
        y = (256 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # 转换回numpy数组
        img_array = np.array(img)
        images.append(img_array)
        
        print(f"  创建图片 {i+1}: {img_array.shape}, 背景色: {['红', '绿', '蓝'][i]}")
    
    return images

def compress_image_sequence_demo(images, output_filename="demo_compressed.png"):
    """演示图片序列压缩功能"""
    print(f"\n开始压缩 {len(images)} 张图片...")
    
    # 准备图片数据
    image_data_list = []
    metadata = {
        "image_count": len(images),
        "compression_level": 6,
        "quality": 95,
        "format": "PNG",
        "timestamp": str(np.datetime64('now')),
        "images": []
    }
    
    for i, img_array in enumerate(images):
        # 转换numpy数组为PIL图像
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # 确保图像是3通道RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, 'RGB')
        else:
            pil_img = Image.fromarray(img_array, 'RGB')
        
        # 压缩图像
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG", optimize=True, compress_level=6)
        
        img_data = img_buffer.getvalue()
        compressed_data = zlib.compress(img_data, level=6)
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        image_data_list.append(base64_data)
        
        # 添加元数据
        metadata["images"].append({
            "index": i,
            "size": len(img_data),
            "compressed_size": len(compressed_data),
            "dimensions": pil_img.size,
            "mode": pil_img.mode
        })
        
        print(f"  压缩图片 {i+1}: 原始大小 {len(img_data)} bytes, 压缩后 {len(compressed_data)} bytes")
    
    # 创建包含所有数据的字典
    combined_data = {
        "metadata": metadata,
        "images": image_data_list
    }
    
    # 将数据编码为JSON字符串
    json_data = json.dumps(combined_data, indent=2)
    
    # 创建最终的压缩图像
    base_img = Image.fromarray(images[0], 'RGB')
    
    # 将JSON数据嵌入到图像中
    final_image = embed_data_in_image(base_img, json_data)
    
    # 保存最终图像
    final_image.save(output_filename, format="PNG", optimize=True)
    
    print(f"压缩完成！输出文件: {output_filename}")
    print(f"JSON数据大小: {len(json_data)} bytes")
    
    return output_filename

def embed_data_in_image(image, data):
    """将数据嵌入到图像中"""
    # 将数据转换为像素值
    data_bytes = data.encode('utf-8')
    data_length = len(data_bytes)
    
    # 计算需要的行数（每行最多255个像素）
    pixels_per_row = 255
    rows_needed = (data_length + pixels_per_row - 1) // pixels_per_row
    
    # 创建新的图像，在底部添加数据行
    new_width = max(image.width, pixels_per_row)
    new_height = image.height + rows_needed + 1  # +1 for length row
    
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_image.paste(image, (0, 0))
    
    # 在底部添加长度信息行
    length_row = [(data_length >> 16) & 255, (data_length >> 8) & 255, data_length & 255]
    for i in range(3):
        if i < new_width:
            new_image.putpixel((i, image.height), (length_row[i], length_row[i], length_row[i]))
    
    # 添加数据行
    for row in range(rows_needed):
        start_idx = row * pixels_per_row
        end_idx = min(start_idx + pixels_per_row, data_length)
        row_data = data_bytes[start_idx:end_idx]
        
        for col, byte_val in enumerate(row_data):
            if col < new_width:
                new_image.putpixel((col, image.height + 1 + row), (byte_val, byte_val, byte_val))
    
    return new_image

def extract_images_demo(compressed_file_path, output_directory="extracted_demo"):
    """演示图片提取功能"""
    print(f"\n开始从 {compressed_file_path} 提取图片...")
    
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
                output_filename = f"extracted_demo_{i+1:04d}.png"
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
    """主函数"""
    print("=== ComfyUI Image Sequence Compressor 演示 ===\n")
    
    # 1. 创建演示图片
    demo_images = create_demo_images()
    
    # 2. 压缩图片序列
    compressed_file = compress_image_sequence_demo(demo_images)
    
    # 3. 提取图片序列
    extract_images_demo(compressed_file)
    
    print("\n=== 演示完成 ===")
    print(f"压缩文件: {compressed_file}")
    print("提取的图片保存在: extracted_demo/ 目录")
    print("\n你可以使用以下命令来提取图片:")
    print(f"python extract_images.py {compressed_file} output_directory")

if __name__ == "__main__":
    main()
