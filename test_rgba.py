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

def test_rgba_conversion():
    # 打开PNG图片
    img = Image.open('test4.png')
    img_array = np.array(img)
    print(f"原始图片尺寸: {img_array.shape}")
    
    # 检查通道数
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("检测到RGBA格式，进行转换...")
        
        # 获取RGB和Alpha通道
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4]
        
        print(f"RGB通道范围: {rgb.min()} - {rgb.max()}")
        print(f"Alpha通道范围: {alpha.min()} - {alpha.max()}")
        
        # 方法1: 直接丢弃Alpha通道
        rgb_direct = rgb.copy()
        print(f"直接转换后尺寸: {rgb_direct.shape}")
        
        # 方法2: 使用Alpha通道混合（更准确）
        alpha_normalized = alpha.astype(np.float32) / 255.0
        rgb_blended = (rgb.astype(np.float32) * alpha_normalized + 
                      (1 - alpha_normalized) * 255).astype(np.uint8)
        print(f"Alpha混合后尺寸: {rgb_blended.shape}")
        
        # 保存转换后的图片进行对比
        Image.fromarray(rgb_direct).save('test4_rgb_direct.png')
        Image.fromarray(rgb_blended).save('test4_rgb_blended.png')
        
        print("转换完成！")
        print("- test4_rgb_direct.png: 直接丢弃Alpha通道")
        print("- test4_rgb_blended.png: Alpha通道混合")
        
        # 测试解码逻辑
        print("\n=== 测试解码逻辑 ===")
        
        # 测试直接转换的图片
        print("测试直接转换的图片:")
        binary_data_direct = extract_binary_from_lsb(rgb_direct)
        if binary_data_direct:
            print(f"提取到二进制数据长度: {len(binary_data_direct)} 位")
            if len(binary_data_direct) >= 32:
                length_binary = binary_data_direct[:32]
                try:
                    data_length = int(length_binary, 2)
                    print(f"数据长度标记: {data_length} 字节")
                except ValueError:
                    print("无法解析数据长度")
        else:
            print("无法提取二进制数据")
        
        # 测试Alpha混合的图片
        print("\n测试Alpha混合的图片:")
        binary_data_blended = extract_binary_from_lsb(rgb_blended)
        if binary_data_blended:
            print(f"提取到二进制数据长度: {len(binary_data_blended)} 位")
            if len(binary_data_blended) >= 32:
                length_binary = binary_data_blended[:32]
                try:
                    data_length = int(length_binary, 2)
                    print(f"数据长度标记: {data_length} 字节")
                except ValueError:
                    print("无法解析数据长度")
        else:
            print("无法提取二进制数据")
        
        return rgb_blended
    else:
        print("图片不是RGBA格式")
        return None

if __name__ == "__main__":
    result = test_rgba_conversion()
