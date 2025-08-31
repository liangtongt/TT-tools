import os
from PIL import Image
import numpy as np
import torch
import hashlib
from typing import List

class TTImgDecPwNode:
    def __init__(self):
        # 使用ComfyUI的默认output目录
        # ComfyUI通常将output目录放在其主目录下
        import folder_paths
        
        # 获取ComfyUI的output目录
        try:
            # 尝试从folder_paths获取output目录
            if hasattr(folder_paths, 'get_output_directory'):
                self.output_dir = folder_paths.get_output_directory()
            elif hasattr(folder_paths, 'output_directory'):
                self.output_dir = folder_paths.output_directory
            else:
                # 如果无法获取，使用默认路径
                self.output_dir = "output"
        except Exception as e:
            print(f"无法获取ComfyUI output目录: {e}")
            self.output_dir = "output"
        
        # 确保目录存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"创建output目录失败: {e}")
            # 如果创建失败，使用当前目录下的output
            self.output_dir = "output"
            os.makedirs(self.output_dir, exist_ok=True)
        

    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "password": ("STRING", {"default": "", "multiline": False}),
                "output_filename": ("STRING", {"default": "tt_img_dec_pw_file", "multiline": False}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "用于解码带密码保护的 tt img enc pw 加密的图片\n需要输入正确的密码才能提取文件\n自动保存到ComfyUI默认output目录\n兼容被RH添加水印的图片\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ()  # 无输出
    FUNCTION = "extract_file_from_image"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def extract_file_from_image(self, image, password="", output_filename="tt_img_dec_pw_file", usage_notes=None):
        """
        从带密码保护的造点图片中提取隐藏文件
        """
        try:
            # 将ComfyUI的torch张量转换为numpy数组
            if hasattr(image, 'cpu'):
                # 如果是torch张量，转换为numpy
                img_np = image.cpu().numpy()
                # 确保值范围在0-255
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                # 如果已经是numpy数组
                img_np = np.array(image).astype(np.uint8)
            
            # 如果是batch，取第一张图片
            if len(img_np.shape) == 4:
                img_np = img_np[0]
            
            # 从图片中提取文件数据
            file_data, file_extension = self._extract_file_data_from_image(img_np, password)
            
            if file_data is None:
                return ()
            
            # 确定输出路径
            if not output_filename:
                output_filename = "tt_img_dec_pw_file"
            
            # 添加扩展名
            if not output_filename.endswith(f".{file_extension}"):
                output_filename = f"{output_filename}.{file_extension}"
            
            # 检查文件名冲突，自动重命名
            base_name = os.path.splitext(output_filename)[0]
            extension = os.path.splitext(output_filename)[1]
            counter = 1
            final_filename = output_filename
            
            while os.path.exists(os.path.join(self.output_dir, final_filename)):
                final_filename = f"{base_name}_{counter}{extension}"
                counter += 1
            
            output_path = os.path.join(self.output_dir, final_filename)
            
            # 保存文件
            with open(output_path, 'wb') as f:
                f.write(file_data)
            
            return ()
            
        except Exception as e:
            return ()
    
    def _extract_file_data_from_image(self, image_array: np.ndarray, password: str = "") -> tuple:
        """
        从图片数组中提取文件数据（支持密码保护）
        
        Args:
            image_array: 图片数组
            password: 密码字符串
        
        Returns:
            tuple: (file_data, file_extension) 或 (None, None)
        """
        try:
            # 支持3通道RGB和4通道RGBA格式
            if len(image_array.shape) != 3 or image_array.shape[2] not in [3, 4]:
                return None, None
            
            height, width, channels = image_array.shape
            
            # 如果是RGBA格式，转换为RGB（丢弃透明度通道）
            if channels == 4:
                # 转换为RGB（丢弃透明度通道）
                image_array = image_array[:, :, :3]
                channels = 3
            
            # 从LSB中提取二进制数据
            binary_data = self._extract_binary_from_lsb(image_array)
            
            if binary_data is None:
                return None, None
            
            # 解析数据长度（前32位）
            if len(binary_data) < 32:
                return None, None
            
            length_binary = binary_data[:32]
            try:
                data_length = int(length_binary, 2)
            except ValueError:
                return None, None
            
            # 检查数据完整性
            expected_bits = 32 + data_length * 8
            if len(binary_data) < expected_bits:
                return None, None
            
            # 提取文件头数据
            file_header_binary = binary_data[32:32 + data_length * 8]
            file_header = self._binary_to_bytes(file_header_binary)
            
            # 解析带密码保护的文件头
            file_data, file_extension = self._parse_file_header_with_password(file_header, password)
            
            return file_data, file_extension
            
        except Exception as e:
            return None, None
    
    def _parse_file_header_with_password(self, file_header: bytes, password: str = "") -> tuple:
        """
        解析带密码保护的文件头
        
        Args:
            file_header: 文件头数据
            password: 密码字符串
        
        Returns:
            tuple: (file_data, file_extension) 或 (None, None)
        """
        try:
            if len(file_header) < 1:
                return None, None
            
            # 读取密码保护标志
            has_password = file_header[0] == 1
            
            if has_password:
                # 有密码保护
                if len(file_header) < 1 + 32 + 16 + 1:
                    print("❌ 密码保护文件头数据不完整")
                    return None, None
                
                # 提取密码哈希和盐值
                password_hash = file_header[1:33]  # 32字节密码哈希
                salt = file_header[33:49]          # 16字节盐值
                
                # 验证密码
                if not self._verify_password(password, salt, password_hash):
                    print("❌ 密码错误！无法解密文件")
                    return None, None
                
                print("✓ 密码验证成功")
                
                # 跳过密码相关字段，从第50字节开始
                header_offset = 49
            else:
                # 无密码保护
                header_offset = 1
            
            # 解析扩展名长度
            if len(file_header) < header_offset + 1:
                return None, None
            
            extension_length = file_header[header_offset]
            
            if len(file_header) < header_offset + 1 + extension_length + 4:
                return None, None
            
            # 解析扩展名
            file_extension = file_header[header_offset + 1:header_offset + 1 + extension_length].decode('utf-8')
            
            # 解析数据长度
            data_size = int.from_bytes(file_header[header_offset + 1 + extension_length:header_offset + 1 + extension_length + 4], 'big')
            
            # 提取文件数据
            file_data = file_header[header_offset + 1 + extension_length + 4:]
            
            # 如果有密码保护，需要解密数据
            if has_password:
                file_data = self._decrypt_data(file_data, password, salt)
            
            print(f"文件扩展名: {file_extension}")
            print(f"文件数据大小: {len(file_data)} 字节")
            
            return file_data, file_extension
            
        except Exception as e:
            print(f"❌ 文件头解析失败: {e}")
            return None, None
    
    def _verify_password(self, password: str, salt: bytes, stored_hash: bytes) -> bool:
        """
        验证密码是否正确
        
        Args:
            password: 输入的密码
            salt: 盐值
            stored_hash: 存储的密码哈希
        
        Returns:
            bool: 密码是否正确
        """
        try:
            # 使用相同的算法生成哈希
            password_hash = hashlib.sha256((password + salt.hex()).encode('utf-8')).digest()
            
            # 比较哈希值
            return password_hash == stored_hash
            
        except Exception as e:
            print(f"❌ 密码验证失败: {e}")
            return False
    
    def _decrypt_data(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        """
        使用密码和盐值解密数据
        
        Args:
            encrypted_data: 加密的数据
            password: 密码
            salt: 盐值
        
        Returns:
            bytes: 解密后的数据
        """
        try:
            # 生成密钥流
            key_stream = self._generate_key_stream(password, salt, len(encrypted_data))
            
            # XOR解密（XOR加密是对称的）
            decrypted = bytearray()
            for i, byte in enumerate(encrypted_data):
                decrypted.append(byte ^ key_stream[i])
            
            return bytes(decrypted)
            
        except Exception as e:
            print(f"❌ 数据解密失败: {e}")
            return b''
    
    def _generate_key_stream(self, password: str, salt: bytes, length: int) -> bytes:
        """
        生成密钥流（与编码节点保持一致）
        
        Args:
            password: 密码
            salt: 盐值
            length: 需要的长度
        
        Returns:
            bytes: 密钥流
        """
        try:
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
            
        except Exception as e:
            print(f"❌ 密钥流生成失败: {e}")
            return b''
    
    def _extract_binary_from_lsb(self, image_array: np.ndarray) -> str:
        """
        从图片的LSB中提取二进制数据（从水印区域后开始，避开水印区域）
        
        Args:
            image_array: 图片数组
        
        Returns:
            str: 二进制字符串
        """
        try:
            height, width, channels = image_array.shape
            watermark_height = int(height * 0.05)  # 水印区域高度为图片高度的5%
            binary_data = ""
            
            # 从水印区域后开始，从每个像素的LSB中提取数据
            for i in range(watermark_height, height):  # 从水印区域后开始
                for j in range(width):
                    for k in range(channels):
                        # 提取最低位
                        bit = image_array[i, j, k] & 1
                        binary_data += str(bit)
                        
                        # 检查是否达到足够的数据长度
                        if len(binary_data) >= 32:  # 至少需要32位来读取长度
                            # 尝试读取长度
                            length_binary = binary_data[:32]
                            try:
                                data_length = int(length_binary, 2)
                                
                                # 计算总需要的位数：32位长度 + 数据长度*8位
                                total_bits_needed = 32 + data_length * 8
                                
                                # 继续提取直到获得完整数据
                                while len(binary_data) < total_bits_needed:
                                    # 计算下一个像素位置
                                    current_pos = len(binary_data)
                                    pixel_index = current_pos // 3
                                    channel_index = current_pos % 3
                                    
                                    # 计算在可用区域中的位置
                                    available_pixels = (height - watermark_height) * width
                                    if pixel_index >= available_pixels:
                                        # 超出可用区域范围，停止提取
                                        break
                                    
                                    # 计算实际的行列位置（加上水印区域偏移）
                                    row = watermark_height + (pixel_index // width)
                                    col = pixel_index % width
                                    
                                    if row < height and col < width:
                                        bit = image_array[row, col, channel_index] & 1
                                        binary_data += str(bit)
                                    else:
                                        break
                                
                                # 如果获得了足够的数据，返回
                                if len(binary_data) >= total_bits_needed:
                                    return binary_data[:total_bits_needed]
                                    
                            except ValueError:
                                # 长度解析失败，继续提取
                                pass
            
            return binary_data
            
        except Exception as e:
            return None
    
    def _binary_to_bytes(self, binary_string: str) -> bytes:
        """
        将二进制字符串转换为字节
        
        Args:
            binary_string: 二进制字符串
        
        Returns:
            bytes: 字节数据
        """
        try:
            # 确保二进制字符串长度是8的倍数
            if len(binary_string) % 8 != 0:
                binary_string = binary_string[:-(len(binary_string) % 8)]
            
            # 转换为字节
            byte_data = bytearray()
            for i in range(0, len(binary_string), 8):
                byte_str = binary_string[i:i+8]
                byte_val = int(byte_str, 2)
                byte_data.append(byte_val)
            
            return bytes(byte_data)
            
        except Exception as e:
            return b''

# 节点类定义完成
