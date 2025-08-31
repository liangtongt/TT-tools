#!/usr/bin/env python3
"""
TT img enc pw 文件提取工具（密码保护版本）
从带密码保护的造点图片中提取隐藏的文件（MP4/JPG等）
支持水印区域兼容性：自动从水印区域后开始读取数据，跳过左上角5%高度的水印区域
需要输入正确的密码才能提取文件
"""

import os
import sys
import numpy as np
from PIL import Image
import hashlib

def extract_file_from_image(image_path: str, password: str, output_path: str = None) -> bool:
    """
    从带密码保护的图片中提取隐藏文件
    
    Args:
        image_path: 造点图片路径
        password: 密码字符串
        output_path: 输出文件路径（可选）
    
    Returns:
        bool: 是否成功提取
    """
    try:
        print(f"教程：https://b23.tv/RbvaMeW")
        print(f"B站：我是小斯呀")
        print(f"正在从图片中提取隐藏文件: {image_path}")
        print(f"密码保护模式: 已启用")
        
        # 读取图片
        image = Image.open(image_path)
        image_array = np.array(image)
        
        print(f"图片尺寸: {image_array.shape}")
        
        # 从图片中提取文件数据
        file_data, file_extension = extract_file_data_from_image(image_array, password)
        
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

def extract_file_data_from_image(image_array: np.ndarray, password: str) -> tuple:
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
            print("❌ 图片必须是3通道RGB或4通道RGBA格式")
            return None, None
        
        height, width, channels = image_array.shape
        
        # 如果是RGBA格式，转换为RGB（丢弃透明度通道）
        if channels == 4:
            # 转换为RGB（丢弃透明度通道）
            image_array = image_array[:, :, :3]
            channels = 3
            print(f"转换后图片尺寸: {image_array.shape}")
        
        # 从LSB中提取二进制数据
        print(f"开始提取数据，水印区域高度: {int(image_array.shape[0] * 0.05)}像素")
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
        
        # 解析带密码保护的文件头
        file_data, file_extension = parse_file_header_with_password(file_header, password)
        
        return file_data, file_extension
        
    except Exception as e:
        print(f"❌ 数据提取失败: {e}")
        return None, None

def parse_file_header_with_password(file_header: bytes, password: str) -> tuple:
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
            if not verify_password(password, salt, password_hash):
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
            file_data = decrypt_data(file_data, password, salt)
        
        print(f"文件扩展名: {file_extension}")
        print(f"文件数据大小: {len(file_data)} 字节")
        
        return file_data, file_extension
        
    except Exception as e:
        print(f"❌ 文件头解析失败: {e}")
        return None, None

def verify_password(password: str, salt: bytes, stored_hash: bytes) -> bool:
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

def decrypt_data(encrypted_data: bytes, password: str, salt: bytes) -> bytes:
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
        key_stream = generate_key_stream(password, salt, len(encrypted_data))
        
        # XOR解密（XOR加密是对称的）
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ key_stream[i])
        
        return bytes(decrypted)
        
    except Exception as e:
        print(f"❌ 数据解密失败: {e}")
        return b''

def generate_key_stream(password: str, salt: bytes, length: int) -> bytes:
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

def extract_binary_from_lsb(image_array: np.ndarray) -> str:
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
    if len(sys.argv) < 3:
        print("使用方法: python tt_img_dec_pw_loc.py <图片路径> <密码> [输出路径]")
        print("示例: python tt_img_dec_pw_loc.py output_image.png mypassword")
        print("示例: python tt_img_dec_pw_loc.py output_image.png mypassword extracted.mp4")
        print("\n注意：此工具用于解码带密码保护的图片，必须提供正确的密码！")
        return
    
    image_path = sys.argv[1]
    password = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    if not password:
        print("❌ 密码不能为空！")
        return
    
    # 提取隐藏文件
    success = extract_file_from_image(image_path, password, output_path)
    
    if success:
        print("\n🎉 隐藏文件提取成功！")
        if output_path:
            print(f"文件位置: {output_path}")
    else:
        print("\n❌ 隐藏文件提取失败！")
        print("请检查:")
        print("1. 图片是否由TT img enc pw节点生成")
        print("2. 密码是否正确")
        print("3. 图片是否完整下载")
        print("4. 图片格式是否正确")
        print("5. 如果图片有水印，工具会自动跳过水印区域（图片高度的5%）")

if __name__ == "__main__":
    main()
