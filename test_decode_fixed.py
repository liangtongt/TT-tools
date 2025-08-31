import numpy as np
from PIL import Image

def extract_binary_from_lsb(image_array):
    """
    从图片的LSB中提取二进制数据（从第51行开始，避开水印区域）
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
                            print(f"检测到数据长度标记: {data_length} 字节")
                            
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
                                print(f"成功提取完整数据: {len(binary_data)} 位")
                                return binary_data[:total_bits_needed]
                            
                        except ValueError:
                            # 长度解析失败，继续提取
                            pass
        
        return binary_data
        
    except Exception as e:
        print(f"❌ LSB提取失败: {e}")
        return None

def binary_to_bytes(binary_string):
    """
    将二进制字符串转换为字节
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

def extract_file_data_from_image(image_array):
    """
    从图片数组中提取文件数据
    """
    try:
        # 支持3通道RGB和4通道RGBA格式
        if len(image_array.shape) != 3 or image_array.shape[2] not in [3, 4]:
            print("❌ 图片必须是3通道RGB或4通道RGBA格式")
            return None, None
        
        height, width, channels = image_array.shape
        
        # 如果是RGBA格式，转换为RGB（丢弃透明度通道）
        if channels == 4:
            print("检测到RGBA格式，自动转换为RGB格式")
            # 使用alpha通道作为权重来混合RGB通道
            alpha = image_array[:, :, 3:4] / 255.0
            rgb = image_array[:, :, :3]
            # 将RGBA转换为RGB，考虑透明度
            image_array = (rgb * alpha + (1 - alpha) * 255).astype(np.uint8)
            channels = 3
            print(f"转换后图片尺寸: {image_array.shape}")
        
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

def main():
    # 打开PNG图片
    img = Image.open('test4.png')
    img_array = np.array(img)
    print(f"原始图片尺寸: {img_array.shape}")
    
    # 提取文件数据
    file_data, file_extension = extract_file_data_from_image(img_array)
    
    if file_data is not None:
        print(f"✓ 成功提取文件数据: {len(file_data)} 字节")
        print(f"文件扩展名: {file_extension}")
        
        # 保存文件
        output_filename = f"extracted_file.{file_extension}"
        with open(output_filename, 'wb') as f:
            f.write(file_data)
        print(f"文件已保存为: {output_filename}")
    else:
        print("❌ 提取失败")

if __name__ == "__main__":
    main()
