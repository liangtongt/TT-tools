import os
from PIL import Image
import numpy as np
import cv2
from typing import List


class TTImgUtils:
    """TT图片处理工具类，包含编码节点的共同方法"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        # 创建必要的目录
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def images_to_mp4(self, images: List[np.ndarray], fps: float) -> str:
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
    
    def image_to_jpg(self, image: np.ndarray, quality: int) -> str:
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
    
    def calculate_required_image_size(self, data: bytes) -> int:
        """计算存储数据所需的图片尺寸（优化存储效率，考虑水印区域）"""
        # 每个像素3个通道，每个通道1位
        bits_needed = len(data) * 8
        pixels_needed = bits_needed // 3
        
        # 考虑水印区域（使用5%的比例，与解码节点保持一致）
        # 先估算一个合理的初始尺寸
        estimated_size = int(np.ceil(np.sqrt(pixels_needed * 1.2)))  # 预留20%空间
        estimated_size = max(64, estimated_size)
        estimated_size = ((estimated_size + 3) // 4) * 4  # 4的倍数对齐
        
        # 使用估算尺寸计算水印区域
        watermark_height = int(estimated_size * 0.05)
        
        # 计算需要的总像素数（包括水印区域）
        # 我们需要确保有足够的像素来存储数据，同时为水印区域预留空间
        total_pixels_needed = pixels_needed + (watermark_height * estimated_size)
        
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
    
    def embed_file_data_in_image(self, image: np.ndarray, file_header: bytes) -> np.ndarray:
        """将文件数据直接嵌入到图片中（不使用ZIP压缩）"""
        # 计算可用区域（使用5%的比例，与解码节点保持一致）
        watermark_height = int(image.shape[0] * 0.05)
        available_height = image.shape[0] - watermark_height
        
        # 检查数据大小是否超过可用图片容量
        max_data_size = available_height * image.shape[1] * 3 // 8  # 每个像素3通道，每8位1字节
        if len(file_header) > max_data_size:
            raise ValueError(f"文件太大 ({len(file_header)} 字节)，当前图片最大支持 {max_data_size} 字节（已排除水印区域）")
        
        print(f"嵌入文件数据: {len(file_header)} 字节到 {image.shape[0]}x{image.shape[1]} 图片（从第{watermark_height+1}行开始，水印区域高度: {watermark_height}像素）")
        
        # 复制图片
        embedded_image = image.copy()
        
        # 将文件数据转换为二进制字符串
        data_binary = ''.join(format(byte, '08b') for byte in file_header)
        
        # 添加数据长度标记（前32位）
        data_length = len(file_header)
        length_binary = format(data_length, '032b')
        full_binary = length_binary + data_binary
        
        # 嵌入数据到图片的LSB（从水印区域后开始，避开水印区域）
        data_index = 0
        for i in range(watermark_height, image.shape[0]):  # 从水印区域后开始
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
    
    def create_storage_image(self, size: int) -> np.ndarray:
        """创建存储图片（纯色背景，最大存储效率）"""
        # 创建纯色背景（灰色，便于LSB隐写）
        image = np.ones((size, size, 3), dtype=np.uint8) * 128
        
        return image
    
    def create_error_image(self, size: int = 512, error_type: str = "enc") -> np.ndarray:
        """创建错误提示图片"""
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # 根据图片尺寸调整文字大小和位置
        scale = size / 512.0
        font_scale = max(0.5, scale)
        thickness = max(1, int(scale))
        
        # 计算文字位置
        text1_pos = (int(50 * scale), int(200 * scale))
        text2_pos = (int(50 * scale), int(250 * scale))
        
        # 根据错误类型设置不同的错误文字
        if error_type == "enc_pw":
            error_text = "Error in TT img enc pw"
        else:
            error_text = "Error in TT img enc"
        
        # 添加错误文字
        cv2.putText(image, error_text, text1_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(image, "Check console for details", text2_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 0, 255), thickness)
        
        return image
    
    def create_storage_image_with_file(self, file_path: str, file_extension: str, file_header: bytes) -> np.ndarray:
        """创建存储图片并嵌入文件（通用版本）"""
        # 计算所需的图片尺寸
        required_size = self.calculate_required_image_size(file_header)
        print(f"文件大小: {len(file_header)} 字节，需要图片尺寸: {required_size}x{required_size}")
        
        # 创建纯色存储图片（最小尺寸，最大存储效率）
        storage_image = self.create_storage_image(required_size)
        
        # 将文件数据直接嵌入到图片中
        embedded_image = self.embed_file_data_in_image(storage_image, file_header)
        
        return embedded_image
