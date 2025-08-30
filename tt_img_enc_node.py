import os
import tempfile
import zipfile
import base64
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
from typing import List, Union, Tuple
import io

class TTImgEncNode:
    def __init__(self):
        self.output_dir = "output"
        self.temp_dir = "temp"
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "noise_density": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5}),
                "noise_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def process_images(self, images, fps=30, quality=95, noise_density=0.1, noise_size=2):
        """
        处理输入的图片，根据数量自动转换格式并嵌入造点图片
        """
        try:
            # 获取图片数量
            num_images = len(images)
            
            # 创建临时文件
            temp_file = None
            file_extension = None
            
            if num_images > 1:
                # 多张图片，转换为MP4
                temp_file = self._images_to_mp4(images, fps)
                file_extension = "mp4"
            else:
                # 单张图片，转换为JPG
                temp_file = self._image_to_jpg(images[0], quality)
                file_extension = "jpg"
            
            # 创建造点图片并嵌入文件
            output_image = self._create_noise_image_with_file(temp_file, file_extension, noise_density, noise_size)
            
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 转换为torch张量，确保与ComfyUI兼容
            output_tensor = torch.from_numpy(output_image).float() / 255.0
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT img enc node: {str(e)}")
            # 返回一个错误提示图片
            error_image = self._create_error_image()
            error_tensor = torch.from_numpy(error_image).float() / 255.0
            return (error_tensor,)
    
    def _images_to_mp4(self, images: List[np.ndarray], fps: int) -> str:
        """将多张图片转换为MP4视频"""
        temp_path = os.path.join(self.temp_dir, "temp_video.mp4")
        
        # 获取图片尺寸
        height, width = images[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        try:
            for img in images:
                # 确保图片是BGR格式（OpenCV要求）
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # RGB转BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    # 灰度图转BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # 确保图片是uint8类型
                if img_bgr.dtype != np.uint8:
                    img_bgr = (img_bgr * 255).astype(np.uint8)
                
                out.write(img_bgr)
        finally:
            out.release()
        
        return temp_path
    
    def _image_to_jpg(self, image: np.ndarray, quality: int) -> str:
        """将单张图片转换为JPG格式"""
        temp_path = os.path.join(self.temp_dir, "temp_image.jpg")
        
        # 转换为PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L').convert('RGB')
        
        # 保存为JPG
        pil_image.save(temp_path, 'JPEG', quality=quality)
        
        return temp_path
    
    def _create_noise_image_with_file(self, file_path: str, file_extension: str, noise_density: float, noise_size: int) -> np.ndarray:
        """创建造点图片并嵌入文件"""
        # 读取文件内容
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 创建ZIP文件
        zip_data = self._create_zip_with_file(file_data, file_extension)
        
        # 将ZIP数据转换为base64字符串
        zip_base64 = base64.b64encode(zip_data).decode('utf-8')
        
        # 创建造点图片
        image_size = 512  # 固定尺寸
        noise_image = self._create_noise_image(image_size, noise_density, noise_size)
        
        # 将base64数据嵌入到图片中
        embedded_image = self._embed_data_in_image(noise_image, zip_base64)
        
        return embedded_image
    
    def _create_zip_with_file(self, file_data: bytes, file_extension: str) -> bytes:
        """创建包含文件的ZIP数据"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            filename = f"output.{file_extension}"
            zip_file.writestr(filename, file_data)
        
        return zip_buffer.getvalue()
    
    def _create_noise_image(self, size: int, density: float, noise_size: int) -> np.ndarray:
        """创建造点图片"""
        # 创建白色背景
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # 添加随机噪点
        num_noise = int(size * size * density)
        
        for _ in range(num_noise):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            
            # 随机颜色
            color = np.random.randint(0, 256, 3)
            
            # 绘制噪点
            cv2.circle(image, (x, y), noise_size, color.tolist(), -1)
        
        return image
    
    def _embed_data_in_image(self, image: np.ndarray, data: str) -> np.ndarray:
        """将数据嵌入到图片中（使用LSB隐写术）"""
        # 将数据转换为二进制
        data_binary = ''.join(format(ord(char), '08b') for char in data)
        
        # 添加结束标记
        data_binary += '00000000'  # 8个0作为结束标记
        
        # 确保图片有足够的像素来存储数据
        required_pixels = len(data_binary)
        if required_pixels > image.shape[0] * image.shape[1] * 3:
            raise ValueError("图片太小，无法存储数据")
        
        # 复制图片
        embedded_image = image.copy()
        
        # 嵌入数据到LSB
        data_index = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):  # RGB通道
                    if data_index < len(data_binary):
                        # 修改最低位
                        if data_binary[data_index] == '1':
                            embedded_image[i, j, k] |= 1
                        else:
                            embedded_image[i, j, k] &= 0xFE
                        data_index += 1
                    else:
                        break
                if data_index >= len(data_binary):
                    break
            if data_index >= len(data_binary):
                break
        
        return embedded_image
    
    def _create_error_image(self) -> np.ndarray:
        """创建错误提示图片"""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # 添加错误文字
        cv2.putText(image, "Error in TT img enc", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, "Check console for details", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return image

# 节点类定义完成
