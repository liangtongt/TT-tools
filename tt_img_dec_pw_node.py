import os
from PIL import Image
import numpy as np
import torch
import hashlib
from typing import List, Tuple, Optional
import cv2
import tempfile
import uuid

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
                "usage_notes": ("STRING", {"default": "通用解码节点：支持带密码保护和无密码的图片\n自动检测图片类型，无需手动选择\n如果图片有密码保护，需要输入正确密码\n如果图片无密码保护，密码字段可留空\n自动保存到ComfyUI默认output目录\n支持直接输出解码后的图片和音频\n图片格式：PNG、JPG、JPEG、BMP、TIFF、WEBP\n视频格式：MP4、AVI、MOV、MKV、WEBM（输出所有帧+音频+FPS）\n音频格式：WAV、MP3、AAC、FLAC、OGG、M4A\n输出：IMAGE(所有帧)、AUDIO、文件路径、FPS\n兼容被RH添加水印的图片\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "FLOAT")  # 图片、音频、文件路径、FPS输出
    RETURN_NAMES = ("image", "audio", "file_path", "fps")
    FUNCTION = "extract_file_from_image"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def extract_file_from_image(self, image, password="", output_filename="tt_img_dec_pw_file", usage_notes=None):
        """
        通用解码函数：支持带密码保护和无密码的造点图片
        自动检测图片类型并选择相应的解码方式
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
                return (None, None, "", 0.0)
            
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
            
            # 处理解码后的文件，转换为ComfyUI格式
            output_image, output_audio, fps = self._process_decoded_file(output_path, file_extension)
            
            return (output_image, output_audio, output_path, fps)
            
        except Exception as e:
            print(f"解码失败: {e}")
            return (None, None, "", 0.0)
    
    def _extract_file_data_from_image(self, image_array: np.ndarray, password: str = "") -> tuple:
        """
        从图片数组中提取文件数据（支持密码保护和无密码）
        自动检测图片类型并选择相应的解码方式
        
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
            
            # 首先尝试解析为密码保护格式
            file_data, file_extension = self._parse_file_header_with_password(file_header, password)
            
            # 如果密码保护格式解析失败，尝试解析为普通格式
            if file_data is None:
                print("尝试解析为普通无密码格式...")
                file_data, file_extension = self._parse_file_header_normal(file_header)
            
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
    
    def _parse_file_header_normal(self, file_header: bytes) -> tuple:
        """
        解析普通无密码保护的文件头
        
        Args:
            file_header: 文件头数据
        
        Returns:
            tuple: (file_data, file_extension) 或 (None, None)
        """
        try:
            if len(file_header) < 5:  # 至少需要1字节扩展名长度 + 4字节数据长度
                return None, None
            
            # 解析扩展名长度
            extension_length = file_header[0]
            
            if len(file_header) < 1 + extension_length + 4:
                return None, None
            
            # 解析扩展名
            file_extension = file_header[1:1 + extension_length].decode('utf-8')
            
            # 解析数据长度
            data_size = int.from_bytes(file_header[1 + extension_length:1 + extension_length + 4], 'big')
            
            # 提取文件数据
            file_data = file_header[1 + extension_length + 4:]
            
            print(f"✓ 检测到普通无密码格式")
            print(f"文件扩展名: {file_extension}")
            print(f"文件数据大小: {len(file_data)} 字节")
            
            return file_data, file_extension
            
        except Exception as e:
            print(f"❌ 普通格式文件头解析失败: {e}")
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
    
    def _process_decoded_file(self, file_path: str, file_extension: str) -> Tuple[Optional[torch.Tensor], Optional[dict], float]:
        """
        处理解码后的文件，转换为ComfyUI兼容的格式
        
        Args:
            file_path: 解码后的文件路径
            file_extension: 文件扩展名
        
        Returns:
            tuple: (image_tensor, audio_dict, fps) 或 (None, None, 0.0)
        """
        try:
            # 图片文件处理
            if file_extension.lower() in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']:
                image_result = self._process_image_file(file_path)
                return image_result, None, 0.0
            
            # 视频文件处理
            elif file_extension.lower() in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                return self._process_video_file(file_path)
            
            # 音频文件处理
            elif file_extension.lower() in ['wav', 'mp3', 'aac', 'flac', 'ogg', 'm4a']:
                audio_result = self._process_audio_file(file_path)
                return None, audio_result, 0.0
            
            # 其他文件类型，返回None
            else:
                print(f"不支持的文件类型: {file_extension}")
                return None, None, 0.0
                
        except Exception as e:
            print(f"处理解码文件失败: {e}")
            return None, None, 0.0
    
    def _process_image_file(self, file_path: str) -> Optional[torch.Tensor]:
        """
        处理图片文件，转换为ComfyUI的IMAGE格式
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            torch.Tensor: ComfyUI格式的图片张量
        """
        try:
            # 使用PIL读取图片
            image = Image.open(file_path)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # 添加batch维度并转换为torch张量
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            return img_tensor
            
        except Exception as e:
            print(f"处理图片文件失败: {e}")
            return None
    
    def _process_video_file(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[dict], float]:
        """
        处理视频文件，提取所有帧作为图片序列，提取音频和FPS
        
        Args:
            file_path: 视频文件路径
        
        Returns:
            tuple: (image_tensor, audio_dict, fps)
        """
        try:
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print(f"无法打开视频文件: {file_path}")
                return None, None, 0.0
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: FPS={fps}, 总帧数={frame_count}")
            
            # 读取所有帧
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                print("无法读取视频帧")
                return None, None, 0.0
            
            # 将所有帧合并为一个张量 (batch, height, width, channels)
            frames_array = np.array(frames).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(frames_array)
            
            print(f"成功读取 {len(frames)} 帧，形状: {img_tensor.shape}")
            
            # 提取音频
            audio_dict = self._extract_audio_from_video(file_path)
            
            return img_tensor, audio_dict, fps
            
        except Exception as e:
            print(f"处理视频文件失败: {e}")
            return None, None, 0.0
    
    def _process_audio_file(self, file_path: str) -> Optional[dict]:
        """
        处理音频文件，转换为ComfyUI的AUDIO格式
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            dict: ComfyUI格式的音频数据
        """
        try:
            # 使用soundfile读取音频
            import soundfile as sf
            
            audio_data, sample_rate = sf.read(file_path)
            
            # 确保是2D数组 (samples, channels)
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            # 转换为torch张量
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # 返回ComfyUI音频格式
            return {
                'samples': audio_tensor,
                'sample_rate': sample_rate
            }
            
        except Exception as e:
            print(f"处理音频文件失败: {e}")
            return None
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[dict]:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            dict: ComfyUI格式的音频数据
        """
        try:
            # 使用FFmpeg提取音频
            import subprocess
            import tempfile
            
            # 创建临时音频文件
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # 使用FFmpeg提取音频
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # 不处理视频
                '-acodec', 'pcm_s16le',  # 音频编码
                '-ar', '44100',  # 采样率
                '-ac', '2',  # 声道数
                '-y',  # 覆盖输出文件
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg提取音频失败: {result.stderr}")
                return None
            
            # 读取提取的音频
            audio_dict = self._process_audio_file(temp_audio_path)
            
            # 清理临时文件
            os.unlink(temp_audio_path)
            
            return audio_dict
            
        except Exception as e:
            print(f"从视频提取音频失败: {e}")
            return None

# 节点类定义完成
