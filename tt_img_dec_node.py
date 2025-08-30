import os
from PIL import Image
import numpy as np
import torch
from typing import List

class TTImgDecNode:
    def __init__(self):
        self.temp_dir = "temp"
        
        # 创建必要的目录
        os.makedirs(self.temp_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_filename": ("STRING", {"default": "extracted_file", "multiline": False}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "从造点图片中提取隐藏的文件（MP4/JPG等）\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")  # 返回提取状态和文件路径
    FUNCTION = "extract_file_from_image"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True
    
    def extract_file_from_image(self, image, output_filename="extracted_file", usage_notes=None):
        """
        从造点图片中提取隐藏文件
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
            
            print(f"正在从图片中提取隐藏文件...")
            print(f"图片尺寸: {img_np.shape}")
            
            # 从图片中提取文件数据
            file_data, file_extension = self._extract_file_data_from_image(img_np)
            
            if file_data is None:
                error_msg = "无法从图片中提取文件数据"
                print(f"❌ {error_msg}")
                if usage_notes:
                    print(f"=== 提取失败，请参考使用说明 ===")
                    print(usage_notes)
                return (error_msg, "")
            
            print(f"✓ 成功提取文件数据: {len(file_data)} 字节")
            print(f"文件扩展名: {file_extension}")
            
            # 确定输出路径
            if not output_filename:
                output_filename = "extracted_file"
            
            # 添加扩展名
            if not output_filename.endswith(f".{file_extension}"):
                output_filename = f"{output_filename}.{file_extension}"
            
            output_path = os.path.join(self.temp_dir, output_filename)
            
            # 保存文件
            with open(output_path, 'wb') as f:
                f.write(file_data)
            
            print(f"✓ 文件已保存到: {output_path}")
            
            # 如果有使用说明，在控制台输出
            if usage_notes:
                print(f"=== TT img dec 使用说明 ===")
                print(usage_notes)
                print(f"=== 提取完成 ===")
                print(f"输出文件: {output_path}")
                print(f"文件大小: {len(file_data)} 字节")
            
            return ("提取成功", output_path)
            
        except Exception as e:
            error_msg = f"提取失败: {str(e)}"
            print(f"❌ {error_msg}")
            if usage_notes:
                print(f"=== 提取失败，但请参考使用说明 ===")
                print(usage_notes)
            return (error_msg, "")
    
    def _extract_file_data_from_image(self, image_array: np.ndarray) -> tuple:
        """
        从图片数组中提取文件数据
        
        Args:
            image_array: 图片数组
        
        Returns:
            tuple: (file_data, file_extension) 或 (None, None)
        """
        try:
            # 确保图片是3通道RGB
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                print("❌ 图片必须是3通道RGB格式")
                return None, None
            
            height, width, channels = image_array.shape
            
            # 从LSB中提取二进制数据
            binary_data = self._extract_binary_from_lsb(image_array)
            
            if binary_data is None:
                return None, None
            
            # 解析数据长度（前32位）
            if len(binary_data) < 32:
                print("❌ 数据长度不足")
                return None, None
            
            length_binary = binary_data[:32]
            try:
                data_length = int(length_binary, 2)
            except ValueError:
                print("❌ 无法解析数据长度")
                return None, None
            
            print(f"数据长度标记: {data_length} 字节")
            
            # 检查数据完整性
            expected_bits = 32 + data_length * 8
            if len(binary_data) < expected_bits:
                print(f"❌ 数据不完整，期望 {expected_bits} 位，实际 {len(binary_data)} 位")
                return None, None
            
            # 提取文件头数据
            file_header_binary = binary_data[32:32 + data_length * 8]
            file_header = self._binary_to_bytes(file_header_binary)
            
            # 解析文件头
            if len(file_header) < 5:  # 至少需要1字节扩展名长度 + 4字节数据长度
                print("❌ 文件头数据不完整")
                return None, None
            
            # 解析扩展名长度
            extension_length = file_header[0]
            
            if len(file_header) < 1 + extension_length + 4:
                print("❌ 文件头数据不完整")
                return None, None
            
            # 解析扩展名
            file_extension = file_header[1:1 + extension_length].decode('utf-8')
            
            # 解析数据长度
            data_size = int.from_bytes(file_header[1 + extension_length:1 + extension_length + 4], 'big')
            
            # 提取文件数据
            file_data = file_header[1 + extension_length + 4:]
            
            print(f"文件扩展名: {file_extension}")
            print(f"文件数据大小: {len(file_data)} 字节")
            
            return file_data, file_extension
            
        except Exception as e:
            print(f"❌ 数据提取失败: {e}")
            return None, None
    
    def _extract_binary_from_lsb(self, image_array: np.ndarray) -> str:
        """
        从图片的LSB中提取二进制数据
        
        Args:
            image_array: 图片数组
        
        Returns:
            str: 二进制字符串
        """
        try:
            height, width, channels = image_array.shape
            binary_data = ""
            
            # 从每个像素的LSB中提取数据
            for i in range(height):
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
                                total_bits_needed = 32 + data_length * 8
                                
                                # 继续提取直到获得完整数据
                                while len(binary_data) < total_bits_needed:
                                    # 计算下一个像素位置
                                    current_pos = len(binary_data)
                                    pixel_index = current_pos // 3
                                    channel_index = current_pos % 3
                                    
                                    if pixel_index >= height * width:
                                        # 超出图片范围，停止提取
                                        break
                                    
                                    row = pixel_index // width
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
            print(f"❌ LSB提取失败: {e}")
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
            print(f"❌ 二进制转换失败: {e}")
            return b''

# 节点类定义完成
