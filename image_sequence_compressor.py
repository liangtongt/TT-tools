import os
import json
import base64
import numpy as np
from PIL import Image
import io
import zlib
from typing import List, Dict, Any, Optional
import torch
import cv2
import tempfile

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
                "quality": ("INT", {"default": 85, "min": 1, "max": 100}),
                "container_size": ("INT", {"default": 512, "min": 256, "max": 2048})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compress_sequence"
    CATEGORY = "image/compression"
    
    def compress_sequence(self, images, quality, container_size):
        """将图片序列压缩并嵌入到自动生成的承载图像中"""
        
        # 准备元数据
        metadata = {
            "image_count": len(images),
            "quality": quality,
            "timestamp": str(np.datetime64('now')),
            "type": "single" if len(images) == 1 else "sequence"
        }
        
        # 根据图片数量选择处理方式
        if len(images) == 1:
            # 单张图片：编码成JPEG
            compressed_data = self._compress_single_image(images[0], quality)
            metadata["type"] = "single"
            metadata["format"] = "JPEG"
        else:
            # 多张图片：编码成MP4
            compressed_data = self._compress_image_sequence(images, quality)
            metadata["type"] = "sequence"
            metadata["format"] = "MP4"
        
        # 将压缩数据编码为base64
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        # 创建包含所有数据的字典
        combined_data = {
            "metadata": metadata,
            "data": base64_data
        }
        
        # 将数据编码为JSON字符串
        json_data = json.dumps(combined_data, indent=2)
        

        
        # 创建承载图像
        container_image = self._create_container_image(container_size, container_size)
        
        # 将JSON数据嵌入到图像中
        final_image = self._embed_data_in_image(container_image, json_data)
        
        # 转换为Tensor格式返回，确保是CHW格式
        final_array = np.array(final_image).astype(np.float32) / 255.0
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
    
    def _compress_single_image(self, img_array, quality):
        """压缩单张图片为JPEG格式"""
        # 处理Tensor对象，转换为numpy数组
        if torch.is_tensor(img_array):
            img_array = img_array.cpu().numpy()
            # 处理批次维度
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            # Tensor通常是CHW格式，需要转换为HWC格式
            if len(img_array.shape) == 3 and img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
        else:
            # 处理numpy数组的批次维度
            if len(img_array.shape) == 4:
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
        
        # 压缩为JPEG
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="JPEG", quality=quality, optimize=True)
        return img_buffer.getvalue()
    
    def _compress_image_sequence(self, images, quality):
        """压缩图片序列为MP4格式"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 获取第一张图片的尺寸
            first_img = self._tensor_to_numpy(images[0])
            height, width = first_img.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))
            
            # 写入每一帧
            for img_array in images:
                # 转换为numpy数组
                img_np = self._tensor_to_numpy(img_array)
                
                # 确保是BGR格式（OpenCV需要）
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_np
                
                out.write(img_bgr)
            
            out.release()
            
            # 读取压缩后的MP4文件
            with open(temp_path, 'rb') as f:
                mp4_data = f.read()
            
            return mp4_data
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _tensor_to_numpy(self, img_array):
        """将Tensor或numpy数组转换为标准numpy数组"""
        # 处理Tensor对象，转换为numpy数组
        if torch.is_tensor(img_array):
            img_array = img_array.cpu().numpy()
            # 处理批次维度
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            # Tensor通常是CHW格式，需要转换为HWC格式
            if len(img_array.shape) == 3 and img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
        else:
            # 处理numpy数组的批次维度
            if len(img_array.shape) == 4:
                img_array = img_array[0]
        
        # 转换数值范围
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        return img_array
    
    def _create_container_image(self, width, height):
        """创建承载数据的图像"""
        # 创建一个渐变背景的图像
        img = Image.new('RGB', (width, height), (240, 240, 240))
        
        # 添加一些简单的图案，让图像看起来更自然
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(img)
        
        # 添加一些随机的小点
        import random
        random.seed(42)  # 固定种子，确保每次生成相同的图像
        
        for _ in range(50):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            draw.point((x, y), fill=color)
        
        return img

# 节点注册已移至 __init__.py
