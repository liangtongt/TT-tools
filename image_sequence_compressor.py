import os
import json
import base64
import numpy as np
from PIL import Image
import io
import zlib
from typing import List, Dict, Any, Optional

class ImageSequenceCompressor:
    """图片序列压缩器节点"""
    
    def __init__(self):
        self.output_dir = "output"
        self.compression_level = 6
        self.quality = 95
        self.format = "PNG"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "compressed_sequence.png"}),
                "compression_level": ("INT", {"default": 6, "min": 1, "max": 9}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "output_directory": ("STRING", {"default": "output"})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("compressed_file_path", "extraction_script")
    FUNCTION = "compress_sequence"
    CATEGORY = "image/compression"
    
    def compress_sequence(self, images, filename, compression_level, quality, format, include_metadata, output_directory):
        """压缩图片序列到一个文件中"""
        
        # 确保输出目录存在
        os.makedirs(output_directory, exist_ok=True)
        
        # 准备图片数据
        image_data_list = []
        metadata = {
            "image_count": len(images),
            "compression_level": compression_level,
            "quality": quality,
            "format": format,
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
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                pil_img = Image.fromarray(img_array, 'RGBA')
                pil_img = pil_img.convert('RGB')
            else:
                pil_img = Image.fromarray(img_array, 'RGB')
            
            # 压缩图像
            img_buffer = io.BytesIO()
            if format == "PNG":
                pil_img.save(img_buffer, format="PNG", optimize=True, compress_level=compression_level)
            elif format == "JPEG":
                pil_img.save(img_buffer, format="JPEG", quality=quality, optimize=True)
            elif format == "WEBP":
                pil_img.save(img_buffer, format="WEBP", quality=quality, method=compression_level)
            
            img_data = img_buffer.getvalue()
            compressed_data = zlib.compress(img_data, level=compression_level)
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
        
        # 创建包含所有数据的字典
        combined_data = {
            "metadata": metadata,
            "images": image_data_list
        }
        
        # 将数据编码为JSON字符串
        json_data = json.dumps(combined_data, indent=2)
        
        # 创建最终的压缩图像
        # 使用第一张图片作为基础，将JSON数据嵌入到EXIF或其他元数据中
        if len(images) > 0:
            base_img_array = images[0]
            if base_img_array.dtype != np.uint8:
                base_img_array = (base_img_array * 255).astype(np.uint8)
            
            if len(base_img_array.shape) == 3 and base_img_array.shape[2] == 3:
                base_img = Image.fromarray(base_img_array, 'RGB')
            elif len(base_img_array.shape) == 3 and base_img_array.shape[2] == 4:
                base_img = Image.fromarray(base_img_array, 'RGBA')
                base_img = base_img.convert('RGB')
            else:
                base_img = Image.fromarray(base_img_array, 'RGB')
            
            # 将JSON数据嵌入到图像中（使用自定义方法）
            final_image = self._embed_data_in_image(base_img, json_data)
            
            # 保存最终图像
            output_path = os.path.join(output_directory, filename)
            final_image.save(output_path, format=format, optimize=True)
            
            # 创建提取脚本
            extraction_script = self._create_extraction_script(output_path)
            
            return output_path, extraction_script
        else:
            raise ValueError("没有输入图像")
    
    def _embed_data_in_image(self, image, data):
        """将数据嵌入到图像中"""
        # 方法1：在图像底部添加一个小的数据条
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
    
    def _create_extraction_script(self, compressed_file_path):
        """创建Python提取脚本"""
        script_content = f'''#!/usr/bin/env python3
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
            
            print(f"\\n所有图片已成功提取到: {output_directory}")
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
        print("\\n提取完成！")
        sys.exit(0)
    else:
        print("\\n提取失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # 保存脚本到文件
        script_path = os.path.join(os.path.dirname(compressed_file_path), "extract_images.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_path

# 节点注册已移至 __init__.py
