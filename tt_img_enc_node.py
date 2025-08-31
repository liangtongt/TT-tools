import os
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List

class TTImgEncNode:
    def __init__(self):
        self.temp_dir = "temp"
        
        # 创建必要的目录
        os.makedirs(self.temp_dir, exist_ok=True)
       
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 120.0}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "利用图片的像素信息保存视频或图片，配合配套的本地解码软件，即可获取原文件\n即使被RH加了水印也能正常解码\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }
        
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def process_images(self, images, fps=16.0, quality=95, usage_notes=None):
        """
        处理输入的图片，根据数量自动转换格式并嵌入造点图片
        """
        try:
            # 获取图片数量
            num_images = len(images)
            
            # 将ComfyUI的torch张量转换为numpy数组
            numpy_images = []
            for img in images:
                if hasattr(img, 'cpu'):
                    # 如果是torch张量，转换为numpy
                    img_np = img.cpu().numpy()
                    # 确保值范围在0-255
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                else:
                    # 如果已经是numpy数组
                    img_np = np.array(img).astype(np.uint8)
                numpy_images.append(img_np)
            
            # 创建临时文件
            temp_file = None
            file_extension = None
            
            if num_images > 1:
                # 多张图片，转换为MP4
                temp_file = self._images_to_mp4(numpy_images, fps)
                file_extension = "mp4"
            else:
                # 单张图片，转换为JPG
                temp_file = self._image_to_jpg(numpy_images[0], quality)
                file_extension = "jpg"
            
            # 创建存储图片并嵌入文件
            output_image = self._create_storage_image_with_file(temp_file, file_extension)
            
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 转换为torch张量，确保与ComfyUI兼容
            # 注意：ComfyUI期望的是(batch_size, height, width, channels)格式
            # 我们只输出一张图片，所以batch_size=1
            output_tensor = torch.from_numpy(output_image).float() / 255.0
            output_tensor = output_tensor.unsqueeze(0)  # 添加batch维度
            
            # 如果有使用说明，在控制台输出
            if usage_notes:
                print(f"=== TT img enc 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"输出图片尺寸: {output_image.shape[1]}x{output_image.shape[0]}")
                print(f"文件类型: {file_extension}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT img enc node: {str(e)}")
            # 返回一个错误提示图片（使用默认尺寸）
            error_image = self._create_error_image(512)
            error_tensor = torch.from_numpy(error_image).float() / 255.0
            error_tensor = error_tensor.unsqueeze(0)  # 添加batch维度
            
            # 如果有使用说明，在错误时也输出
            if usage_notes:
                print(f"=== 处理失败，但请参考使用说明 ===")
                print(usage_notes)
            
            return (error_tensor,)
    
    def _images_to_mp4(self, images: List[np.ndarray], fps: float) -> str:
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
    
    def _create_storage_image_with_file(self, file_path: str, file_extension: str) -> np.ndarray:
        """创建存储图片并嵌入文件"""
        # 读取文件内容
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 创建文件头信息（包含文件扩展名）
        file_header = self._create_file_header(file_data, file_extension)
        
        # 计算所需的图片尺寸
        required_size = self._calculate_required_image_size(file_header)
        print(f"文件大小: {len(file_header)} 字节，需要图片尺寸: {required_size}x{required_size}")
        
        # 创建纯色存储图片（最小尺寸，最大存储效率）
        storage_image = self._create_storage_image(required_size)
        
        # 将文件数据直接嵌入到图片中
        embedded_image = self._embed_file_data_in_image(storage_image, file_header)
        
        return embedded_image
    

    def _calculate_required_image_size(self, data: bytes) -> int:
        """计算存储数据所需的图片尺寸（优化存储效率，考虑水印区域）"""
        # 每个像素3个通道，每个通道1位
        bits_needed = len(data) * 8
        pixels_needed = bits_needed // 3
        
        # 考虑水印区域（前50行不可用）
        watermark_height = 50
        
        # 计算需要的总像素数（包括水印区域）
        # 我们需要确保有足够的像素来存储数据，同时为水印区域预留空间
        total_pixels_needed = pixels_needed + (watermark_height * int(np.ceil(np.sqrt(pixels_needed))))
        
        # 计算正方形图片的边长
        side_length = int(np.ceil(np.sqrt(total_pixels_needed)))
        
        # 确保最小尺寸为64
        side_length = max(64, side_length)
        
        # 向上取整到最近的4的倍数（更小的对齐，提高效率）
        side_length = ((side_length + 3) // 4) * 4
        
        # 验证计算是否正确
        available_pixels = (side_length - watermark_height) * side_length
        required_pixels = pixels_needed
        
        if available_pixels < required_pixels:
            # 如果可用像素不够，增加图片尺寸
            side_length = int(np.ceil(np.sqrt(required_pixels + watermark_height * side_length)))
            side_length = ((side_length + 3) // 4) * 4
        
        return side_length
    
    def _create_file_header(self, file_data: bytes, file_extension: str) -> bytes:
        """创建文件头信息（包含文件扩展名和数据）"""
        # 文件头格式：[扩展名长度(1字节)][扩展名][数据长度(4字节)][数据]
        extension_bytes = file_extension.encode('utf-8')
        extension_length = len(extension_bytes)
        
        # 检查扩展名长度（最大255字节）
        if extension_length > 255:
            raise ValueError(f"文件扩展名太长: {file_extension}")
        
        # 构建文件头
        header = bytearray()
        header.append(extension_length)  # 扩展名长度
        header.extend(extension_bytes)   # 扩展名
        header.extend(len(file_data).to_bytes(4, 'big'))  # 数据长度（4字节）
        header.extend(file_data)         # 文件数据
        
        return bytes(header)
    
    def _embed_file_data_in_image(self, image: np.ndarray, file_header: bytes) -> np.ndarray:
        """将文件数据直接嵌入到图片中（不使用ZIP压缩）"""
        # 计算可用区域（排除左上角50像素高度的水印区域）
        watermark_height = 50
        available_height = image.shape[0] - watermark_height
        
        # 检查数据大小是否超过可用图片容量
        max_data_size = available_height * image.shape[1] * 3 // 8  # 每个像素3通道，每8位1字节
        if len(file_header) > max_data_size:
            raise ValueError(f"文件太大 ({len(file_header)} 字节)，当前图片最大支持 {max_data_size} 字节（已排除水印区域）")
        
        print(f"嵌入文件数据: {len(file_header)} 字节到 {image.shape[0]}x{image.shape[1]} 图片（从第{watermark_height+1}行开始）")
        
        # 复制图片
        embedded_image = image.copy()
        
        # 将文件数据转换为二进制字符串
        data_binary = ''.join(format(byte, '08b') for byte in file_header)
        
        # 添加数据长度标记（前32位）
        data_length = len(file_header)
        length_binary = format(data_length, '032b')
        full_binary = length_binary + data_binary
        
        # 嵌入数据到图片的LSB（从第51行开始，避开水印区域）
        data_index = 0
        for i in range(watermark_height, image.shape[0]):  # 从第51行开始
            for j in range(image.shape[1]):
                for k in range(3):  # RGB通道
                    if data_index < len(full_binary):
                        # 修改最低位
                        if full_binary[data_index] == '1':
                            embedded_image[i, j, k] |= 1
                        else:
                            embedded_image[i, j, k] &= 0xFE
                        data_index += 1
                    else:
                        break
                if data_index >= len(full_binary):
                    break
            if data_index >= len(full_binary):
                break
        
        return embedded_image
    
    def _create_storage_image(self, size: int) -> np.ndarray:
        """创建存储图片（纯色背景，最大存储效率）"""
        # 创建纯色背景（灰色，便于LSB隐写）
        image = np.ones((size, size, 3), dtype=np.uint8) * 128
        
        return image
    

    
    def _create_error_image(self, size: int = 512) -> np.ndarray:
        """创建错误提示图片"""
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # 根据图片尺寸调整文字大小和位置
        scale = size / 512.0
        font_scale = max(0.5, scale)
        thickness = max(1, int(scale))
        
        # 计算文字位置
        text1_pos = (int(50 * scale), int(200 * scale))
        text2_pos = (int(50 * scale), int(250 * scale))
        
        # 添加错误文字
        cv2.putText(image, "Error in TT img enc", text1_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(image, "Check console for details", text2_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 0, 255), thickness)
        
        return image

# 节点类定义完成
