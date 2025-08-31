#!/usr/bin/env python3
"""
TT img enc 文件提取工具
从造点图片中提取隐藏的文件（MP4/JPG等）
支持水印区域兼容性：自动从第51行开始读取数据，跳过左上角50像素水印区域
"""

import os
import sys
import numpy as np
from PIL import Image

def extract_file_from_image(image_path: str, output_path: str = None) -> bool:
    """
    从图片中提取隐藏文件
    
    Args:
        image_path: 造点图片路径
        output_path: 输出文件路径（可选）
    
    Returns:
        bool: 是否成功提取
    """
    try:
        print(f"正在从图片中提取隐藏文件: {image_path}")
        
        # 读取图片
        image = Image.open(image_path)
        image_array = np.array(image)
        
        print(f"图片尺寸: {image_array.shape}")
        
        # 从图片中提取文件数据
        file_data, file_extension = extract_file_data_from_image(image_array)
        
        if file_data is None:
            print("❌ 无法从图片中提取文件数据")
            return False
        
        print(f"✓ 成功提取文件数据: {len(file_data)} 字节")
        
        # 确定输出路径
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}.{file_extension}"
        
        # 保存文件
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        print(f"✓ 文件已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return False

def extract_file_data_from_image(image_array: np.ndarray) -> tuple:
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
        binary_data = extract_binary_from_lsb(image_array)
        
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
        file_header = binary_to_bytes(file_header_binary)
        
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

def extract_binary_from_lsb(image_array: np.ndarray) -> str:
    """
    从图片的LSB中提取二进制数据（从第51行开始，避开水印区域）
    
    Args:
        image_array: 图片数组
    
    Returns:
        str: 二进制字符串
    """
    try:
        height, width, channels = image_array.shape
        watermark_height = 50  # 水印区域高度
        binary_data = ""
        
        # 从第51行开始，从每个像素的LSB中提取数据
        for i in range(watermark_height, height):  # 从第51行开始
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
                                # 计算下一个像素位置（考虑水印区域偏移）
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
        print(f"❌ LSB提取失败: {e}")
        return None

def binary_to_bytes(binary_string: str) -> bytes:
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

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python extract_zip.py <图片路径> [输出路径]")
        print("示例: python extract_zip.py output_image.png")
        print("示例: python extract_zip.py output_image.png extracted.mp4")
        print("\n💡 支持水印兼容性：自动跳过左上角50像素水印区域")
        return
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    # 提取隐藏文件
    success = extract_file_from_image(image_path, output_path)
    
    if success:
        print("\n🎉 隐藏文件提取成功！")
        if output_path:
            print(f"文件位置: {output_path}")
    else:
        print("\n❌ 隐藏文件提取失败！")
        print("请检查:")
        print("1. 图片是否由TT img enc节点生成")
        print("2. 图片是否完整下载")
        print("3. 图片格式是否正确")
        print("4. 如果图片有水印，工具会自动跳过水印区域")

if __name__ == "__main__":
    main()
