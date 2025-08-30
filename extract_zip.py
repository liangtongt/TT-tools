#!/usr/bin/env python3
"""
TT img enc ZIP文件提取工具
从造点图片中提取隐藏的ZIP文件
"""

import os
import sys
import numpy as np
from PIL import Image
import zipfile
import io

def extract_zip_from_image(image_path: str, output_path: str = None) -> bool:
    """
    从图片中提取ZIP文件
    
    Args:
        image_path: 造点图片路径
        output_path: 输出ZIP文件路径（可选）
    
    Returns:
        bool: 是否成功提取
    """
    try:
        print(f"正在从图片中提取ZIP文件: {image_path}")
        
        # 读取图片
        image = Image.open(image_path)
        image_array = np.array(image)
        
        print(f"图片尺寸: {image_array.shape}")
        
        # 从图片中提取ZIP数据
        zip_data = extract_zip_data_from_image(image_array)
        
        if zip_data is None:
            print("❌ 无法从图片中提取ZIP数据")
            return False
        
        print(f"✓ 成功提取ZIP数据: {len(zip_data)} 字节")
        
        # 确定输出路径
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}.zip"
        
        # 保存ZIP文件
        with open(output_path, 'wb') as f:
            f.write(zip_data)
        
        print(f"✓ ZIP文件已保存到: {output_path}")
        
        # 验证ZIP文件
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                print(f"✓ ZIP文件验证成功，包含文件: {file_list}")
        except Exception as e:
            print(f"⚠️  ZIP文件验证失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return False

def extract_zip_data_from_image(image_array: np.ndarray) -> bytes:
    """
    从图片数组中提取ZIP数据
    
    Args:
        image_array: 图片数组
    
    Returns:
        bytes: ZIP文件数据，如果失败返回None
    """
    try:
        # 确保图片是3通道RGB
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            print("❌ 图片必须是3通道RGB格式")
            return None
        
        height, width, channels = image_array.shape
        
        # 从LSB中提取二进制数据
        binary_data = extract_binary_from_lsb(image_array)
        
        if binary_data is None:
            return None
        
        # 解析数据长度（前32位）
        if len(binary_data) < 32:
            print("❌ 数据长度不足")
            return None
        
        length_binary = binary_data[:32]
        try:
            data_length = int(length_binary, 2)
        except ValueError:
            print("❌ 无法解析数据长度")
            return None
        
        print(f"数据长度标记: {data_length} 字节")
        
        # 检查数据完整性
        expected_bits = 32 + data_length * 8
        if len(binary_data) < expected_bits:
            print(f"❌ 数据不完整，期望 {expected_bits} 位，实际 {len(binary_data)} 位")
            return None
        
        # 提取ZIP数据
        zip_binary = binary_data[32:32 + data_length * 8]
        
        # 转换为字节
        zip_data = binary_to_bytes(zip_binary)
        
        return zip_data
        
    except Exception as e:
        print(f"❌ 数据提取失败: {e}")
        return None

def extract_binary_from_lsb(image_array: np.ndarray) -> str:
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
        print("示例: python extract_zip.py output_image.png extracted.zip")
        return
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    # 提取ZIP文件
    success = extract_zip_from_image(image_path, output_path)
    
    if success:
        print("\n🎉 ZIP文件提取成功！")
        if output_path:
            print(f"文件位置: {output_path}")
    else:
        print("\n❌ ZIP文件提取失败！")
        print("请检查:")
        print("1. 图片是否由TT img enc节点生成")
        print("2. 图片是否完整下载")
        print("3. 图片格式是否正确")

if __name__ == "__main__":
    main()
