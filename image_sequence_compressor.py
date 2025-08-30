import os
import json
import base64
import numpy as np
from PIL import Image
import io
import zlib
from typing import List, Dict, Any, Optional
import torch

class ImageSequenceCompressor:
    """图片序列压缩器节点 - 将图片序列压缩并嵌入到指定图像中"""
    
    def __init__(self):
        self.compression_level = 6
        self.quality = 95
        self.format = "PNG"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "image": ("IMAGE",),
                "compression_level": ("INT", {"default": 6, "min": 1, "max": 9}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "include_metadata": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compress_sequence"
    CATEGORY = "image/compression"
    
    def compress_sequence(self, images, image, compression_level, quality, format, include_metadata):
        """将图片序列压缩并嵌入到指定的图像中"""
        
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
            # 处理Tensor对象，转换为numpy数组
            if torch.is_tensor(img_array):
                img_array = img_array.cpu().numpy()
                # 处理批次维度
                if len(img_array.shape) == 4:
                    # 如果是批次格式 (B, C, H, W)，取第一张图片
                    img_array = img_array[0]
                # Tensor通常是CHW格式，需要转换为HWC格式
                if len(img_array.shape) == 3 and img_array.shape[0] == 3:
                    img_array = np.transpose(img_array, (1, 2, 0))
            else:
                # 处理numpy数组的批次维度
                if len(img_array.shape) == 4:
                    # 如果是批次格式 (B, H, W, C)，取第一张图片
                    img_array = img_array[0]
            
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
        
        # 使用输入的image作为基础图像
        # 处理Tensor对象，转换为numpy数组
        if torch.is_tensor(image):
            base_img_array = image.cpu().numpy()
            # 处理批次维度
            if len(base_img_array.shape) == 4:
                # 如果是批次格式 (B, C, H, W)，取第一张图片
                base_img_array = base_img_array[0]
            # Tensor通常是CHW格式，需要转换为HWC格式
            if len(base_img_array.shape) == 3 and base_img_array.shape[0] == 3:
                base_img_array = np.transpose(base_img_array, (1, 2, 0))
        else:
            base_img_array = image
            # 处理numpy数组的批次维度
            if len(base_img_array.shape) == 4:
                # 如果是批次格式 (B, H, W, C)，取第一张图片
                base_img_array = base_img_array[0]
        
        if base_img_array.dtype != np.uint8:
            base_img_array = (base_img_array * 255).astype(np.uint8)
        else:
            base_img_array = base_img_array
        
        # 确保图像是3通道RGB
        if len(base_img_array.shape) == 3 and base_img_array.shape[2] == 3:
            base_img = Image.fromarray(base_img_array, 'RGB')
        elif len(base_img_array.shape) == 3 and base_img_array.shape[2] == 4:
            base_img = Image.fromarray(base_img_array, 'RGBA')
            base_img = base_img.convert('RGB')
        else:
            base_img = Image.fromarray(base_img_array, 'RGB')
        
        # 将JSON数据嵌入到图像中
        final_image = self._embed_data_in_image(base_img, json_data)
        
        # 转换回numpy数组格式，保持与输入相同的数值范围
        if torch.is_tensor(image):
            # 如果输入是Tensor，检查数值范围
            if image.max() <= 1.0:
                # 如果输入是0-1范围，输出也保持0-1范围
                final_array = np.array(final_image).astype(np.float32) / 255.0
            else:
                # 如果输入是0-255范围，输出也保持0-255范围
                final_array = np.array(final_image).astype(np.uint8)
        else:
            # 如果输入是numpy数组
            if image.dtype != np.uint8:
                # 如果输入是0-1范围，输出也保持0-1范围
                final_array = np.array(final_image).astype(np.float32) / 255.0
            else:
                # 如果输入是0-255范围，输出也保持0-255范围
                final_array = np.array(final_image).astype(np.uint8)
        
        # 转换为Tensor格式返回，确保是CHW格式
        if len(final_array.shape) == 3 and final_array.shape[2] == 3:
            # 如果是HWC格式，转换为CHW格式
            final_array = np.transpose(final_array, (2, 0, 1))
        
        return (torch.from_numpy(final_array),)
    
    def _embed_data_in_image(self, image, data):
        """将数据嵌入到图像中（简单像素替换）"""
        # 将数据转换为像素值
        data_bytes = data.encode('utf-8')
        data_length = len(data_bytes)
        
        # 创建图像副本
        new_image = image.copy()
        width, height = new_image.size
        
        # 检查图像是否有足够的像素来存储数据
        total_pixels = width * height
        required_pixels = data_length + 4  # +4 for length info
        
        if total_pixels < required_pixels:
            # 如果图像太小，扩展图像
            new_width = max(width, int(required_pixels ** 0.5) + 1)
            new_height = max(height, (required_pixels + new_width - 1) // new_width)
            new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            new_image.paste(image, (0, 0))
            width, height = new_image.size
        
        # 将数据长度嵌入到前4个像素中
        length_bytes = data_length.to_bytes(4, byteorder='big')
        for i in range(4):
            x, y = i % width, i // width
            if y < height:
                new_image.putpixel((x, y), (length_bytes[i], length_bytes[i], length_bytes[i]))
        
        # 将数据嵌入到后续像素中
        for i, byte_val in enumerate(data_bytes):
            pixel_index = i + 4  # 跳过长度信息
            x, y = pixel_index % width, pixel_index // width
            if y < height:
                new_image.putpixel((x, y), (byte_val, byte_val, byte_val))
        
        return new_image

# 节点注册已移至 __init__.py
