import os
import numpy as np
import torch
import hashlib
import secrets
from typing import List
try:
    from .tt_img_utils import TTImgUtils
except ImportError:
    from tt_img_utils import TTImgUtils

class TTImgEncPwNode:
    def __init__(self):
        self.temp_dir = "temp"
        self.utils = TTImgUtils(self.temp_dir)
       
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "password": ("STRING", {"default": "", "multiline": False}),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 120.0}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "利用图片的像素信息保存视频或图片，支持密码保护\n配合配套的本地解码软件，即可获取原文件\n即使被RH加了水印也能正常解码\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }
        
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def process_images(self, images, password="", fps=16.0, quality=95, usage_notes=None):
        """
        处理输入的图片，根据数量自动转换格式并嵌入造点图片，支持密码保护
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
                temp_file = self.utils.images_to_mp4(numpy_images, fps)
                file_extension = "mp4"
            else:
                # 单张图片，转换为JPG
                temp_file = self.utils.image_to_jpg(numpy_images[0], quality)
                file_extension = "jpg"
            
            # 创建存储图片并嵌入文件（带密码保护）
            output_image = self._create_storage_image_with_file(temp_file, file_extension, password)
            
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 转换为torch张量，确保与ComfyUI兼容
            output_tensor = torch.from_numpy(output_image).float() / 255.0
            output_tensor = output_tensor.unsqueeze(0)  # 添加batch维度
            
            # 如果有使用说明，在控制台输出
            if usage_notes:
                print(f"=== TT img enc pw 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"输出图片尺寸: {output_image.shape[1]}x{output_image.shape[0]}")
                print(f"文件类型: {file_extension}")
                if password:
                    print(f"密码保护: 已启用")
                else:
                    print(f"密码保护: 未启用")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT img enc pw node: {str(e)}")
            # 返回一个错误提示图片
            error_image = self.utils.create_error_image(512, "enc_pw")
            error_tensor = torch.from_numpy(error_image).float() / 255.0
            error_tensor = error_tensor.unsqueeze(0)
            
            if usage_notes:
                print(f"=== 处理失败，但请参考使用说明 ===")
                print(usage_notes)
            
            return (error_tensor,)
    

    
    def _create_storage_image_with_file(self, file_path: str, file_extension: str, password: str = "") -> np.ndarray:
        """创建存储图片并嵌入文件（支持密码保护）"""
        # 读取文件内容
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 创建文件头信息（包含文件扩展名和密码保护）
        file_header = self._create_file_header_with_password(file_data, file_extension, password)
        
        # 使用共享工具创建存储图片并嵌入文件
        return self.utils.create_storage_image_with_file(file_path, file_extension, file_header)
    


    
    def _create_file_header_with_password(self, file_data: bytes, file_extension: str, password: str = "") -> bytes:
        """创建带密码保护的文件头信息"""
        # 文件头格式：[密码标志(1字节)][密码哈希(32字节)][盐值(16字节)][扩展名长度(1字节)][扩展名][数据长度(4字节)][数据]
        
        # 密码保护标志
        has_password = len(password) > 0
        
        # 构建文件头
        header = bytearray()
        
        if has_password:
            # 有密码保护
            header.append(1)  # 密码标志：1表示有密码
            
            # 生成随机盐值
            salt = secrets.token_bytes(16)
            
            # 使用盐值和密码生成哈希
            password_hash = hashlib.sha256((password + salt.hex()).encode('utf-8')).digest()
            
            header.extend(password_hash)  # 密码哈希（32字节）
            header.extend(salt)           # 盐值（16字节）
            
            # 加密文件数据
            encrypted_data = self._encrypt_data(file_data, password, salt)
            file_data = encrypted_data
        else:
            # 无密码保护
            header.append(0)  # 密码标志：0表示无密码
        
        # 添加扩展名信息
        extension_bytes = file_extension.encode('utf-8')
        extension_length = len(extension_bytes)
        
        # 检查扩展名长度（最大255字节）
        if extension_length > 255:
            raise ValueError(f"文件扩展名太长: {file_extension}")
        
        header.append(extension_length)  # 扩展名长度
        header.extend(extension_bytes)   # 扩展名
        header.extend(len(file_data).to_bytes(4, 'big'))  # 数据长度（4字节）
        header.extend(file_data)         # 文件数据
        
        return bytes(header)
    
    def _encrypt_data(self, data: bytes, password: str, salt: bytes) -> bytes:
        """使用密码和盐值加密数据"""
        # 简单的XOR加密（实际应用中可以使用更安全的加密算法）
        # 使用密码和盐值生成密钥流
        key_stream = self._generate_key_stream(password, salt, len(data))
        
        # XOR加密
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_stream[i])
        
        return bytes(encrypted)
    
    def _generate_key_stream(self, password: str, salt: bytes, length: int) -> bytes:
        """生成密钥流"""
        # 使用密码和盐值生成密钥流
        key_material = (password + salt.hex()).encode('utf-8')
        key_stream = bytearray()
        
        # 使用SHA256生成密钥流
        counter = 0
        while len(key_stream) < length:
            # 组合密码、盐值和计数器
            combined = key_material + str(counter).encode('utf-8')
            hash_result = hashlib.sha256(combined).digest()
            key_stream.extend(hash_result)
            counter += 1
        
        return key_stream[:length]
    


# 节点类定义完成
