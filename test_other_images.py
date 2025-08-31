import numpy as np
from PIL import Image
import os

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
                            # 只显示合理的数据长度（小于1MB）
                            if data_length < 1024 * 1024:
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

def test_image(image_path):
    """测试单个图片"""
    print(f"\n=== 测试图片: {image_path} ===")
    
    try:
        # 打开图片
        img = Image.open(image_path)
        img_array = np.array(img)
        print(f"图片尺寸: {img_array.shape}")
        
        # 检查通道数
        if len(img_array.shape) == 3:
            channels = img_array.shape[2]
            if channels == 4:
                print("RGBA格式，转换为RGB...")
                # 转换为RGB
                alpha = img_array[:, :, 3:4] / 255.0
                rgb = img_array[:, :, :3]
                img_array = (rgb * alpha + (1 - alpha) * 255).astype(np.uint8)
                print(f"转换后尺寸: {img_array.shape}")
            elif channels == 3:
                print("RGB格式")
            else:
                print(f"不支持的通道数: {channels}")
                return
            
            # 提取二进制数据
            binary_data = extract_binary_from_lsb(img_array)
            if binary_data:
                print(f"提取到二进制数据: {len(binary_data)} 位")
            else:
                print("未提取到二进制数据")
        else:
            print("不是3D图片数组")
            
    except Exception as e:
        print(f"处理图片失败: {e}")

def main():
    # 测试所有PNG图片
    image_files = ['test.png', 'test1.png', 'test3.png', 'test4.png']
    
    for image_file in image_files:
        if os.path.exists(image_file):
            test_image(image_file)
        else:
            print(f"图片文件不存在: {image_file}")

if __name__ == "__main__":
    main()
