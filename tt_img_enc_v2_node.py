import os
import numpy as np
import torch
from typing import List
try:
    from .tt_img_utils import TTImgUtils
except ImportError:
    from tt_img_utils import TTImgUtils


class TTImgEncV2Node:
    def __init__(self):
        self.temp_dir = "temp"
        self.utils = TTImgUtils(self.temp_dir)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 120.0}),
                # 视频压缩率（FFmpeg CRF值，18=高质量，23=默认，28=低质量）
                "video_compression": ("INT", {"default": 19, "min": 0, "max": 51}),
                "png_compression": ("INT", {"default": 6, "min": 0, "max": 9}),
                # 跳过上下各20%区域（避免水印/边界干扰），仅使用中间60%写入
                "skip_watermark_area": ("BOOLEAN", {"default": True}),
                
            },
            "optional": {
                "audio": ("AUDIO",),
                "usage_notes": ("STRING", {"default": "V2：更高的存储效率和编解码速度\nvideo_compression(0-51): 视频压缩率，越低质量越高，体积越大，建议默认\npng_compression(0-9):图片压缩率，越低质量越高，体积越大，建议默认 \nskip_watermark_area:是否跳过水印区域，不跳过可以进一步提升存储效率，不确定生成的图片是否会带水印，建议开启\n教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True

    def process_images(self, images, fps=16.0, png_compression=6, skip_watermark_area=True, video_compression=19, audio=None, usage_notes=None):
        """
        将输入图片打包为文件，并以更高位宽的隐写方式写入到存储图片中。
        决策逻辑：
        - 单张图片：PNG格式
        - 2-9张图片：多文件打包（参考hide_v2.js的encodeMultipleFilesToImageV2）
        - 10张及以上：MP4视频（可选音频）
        与V1不同：
        - 使用整张图片区域（不保留水印安全边界）
        - 使用可配置的每通道位数（默认8位）提升容量
        """
        try:
            # 自动使用最大位宽
            bits_per_channel = 8

            num_images = len(images)
            numpy_images = self._batch_convert_images(images)

            temp_file = None
            file_extension = None
            file_count = 1

            if num_images == 1:
                # 单张图片：PNG格式
                temp_file = self.utils.image_to_png(numpy_images[0], png_compression)
                file_extension = "png"
                file_count = 1
            elif num_images < 10:
                # 2-9张图片：多文件打包
                print(f"[V2][ENC] 检测到多文件模式，图片数量: {num_images}")
                # 直接处理多文件数据包，不创建临时文件
                multi_file_data = self._create_multi_file_package_data(numpy_images, png_compression)
                temp_file = None  # 不创建临时文件
                file_extension = "multi"
                file_count = num_images
                print(f"[V2][ENC] 多文件数据包创建完成，总大小: {len(multi_file_data)} 字节")
            else:
                # 10张及以上：MP4视频
                if audio is not None:
                    temp_file = self.utils.images_to_mp4_with_audio(numpy_images, fps, audio, video_compression)
                else:
                    temp_file = self.utils.images_to_mp4(numpy_images, fps, video_compression)
                file_extension = "mp4"
                file_count = num_images

            if file_extension == "multi":
                # 多文件打包：直接使用数据包创建存储图片
                print(f"[V2][ENC] 开始创建多文件存储图片，数据大小: {len(multi_file_data)} 字节")
                output_image = self._create_storage_image_in_memory_v2_multi(multi_file_data, bits_per_channel, skip_watermark_area)
                print(f"[V2][ENC] 多文件存储图片创建完成，尺寸: {output_image.shape}")
            else:
                # 单文件或视频：使用文件路径创建存储图片
                print(f"[V2][ENC] 开始创建单文件存储图片，文件: {temp_file}")
                output_image = self._create_storage_image_in_memory_v2(temp_file, file_extension, bits_per_channel, skip_watermark_area)
                print(f"[V2][ENC] 单文件存储图片创建完成，尺寸: {output_image.shape}")

            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            output_tensor = self._convert_to_tensor(output_image)

            if usage_notes:
                print(f"=== TT img enc V2 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"输出图片尺寸: {output_image.shape[1]}x{output_image.shape[0]}")
                print(f"文件类型: {file_extension}")
                print(f"文件数量: {file_count}")
                print(f"每通道位数: {bits_per_channel}")
                print(f"跳过上下20%: {bool(skip_watermark_area)}")
                if file_extension == "mp4":
                    print(f"视频压缩率(CRF): {video_compression}")

            return (output_tensor,)

        except Exception as e:
            print(f"Error in TT img enc V2 node: {str(e)}")
            error_image = self._create_error_image_v2()
            error_tensor = self._convert_to_tensor(error_image)
            if usage_notes:
                print(f"=== 处理失败，但请参考使用说明 ===")
                print(usage_notes)
            return (error_tensor,)

    def _batch_convert_images(self, images):
        num_images = len(images)
        numpy_images = [None] * num_images
        for i, img in enumerate(images):
            if hasattr(img, 'cpu'):
                img_np = img.cpu().numpy()
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                img_np = np.array(img, dtype=np.uint8)
            numpy_images[i] = img_np
        return numpy_images

    def _convert_to_tensor(self, image):
        if isinstance(image, np.ndarray):
            if torch.cuda.is_available():
                output_tensor = torch.from_numpy(image).cuda().float() / 255.0
            else:
                output_tensor = torch.from_numpy(image).float() / 255.0
        else:
            output_tensor = image.float() / 255.0
        return output_tensor.unsqueeze(0)

    def _create_error_image_v2(self, size: int = 512) -> np.ndarray:
        # 独立于utils，避免影响旧节点
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        import cv2
        scale = size / 512.0
        font_scale = max(0.5, scale)
        thickness = max(1, int(scale))
        cv2.putText(img, "Error in TT img enc V2", (int(40*scale), int(200*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(img, "Check console for details", (int(40*scale), int(250*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
        return img

    def _create_storage_image_in_memory_v2(self, file_path: str, file_extension: str, bits_per_channel: int, skip_watermark_area: bool) -> np.ndarray:
        with open(file_path, 'rb') as f:
            file_data = f.read()

        file_header = self._create_file_header(file_data, file_extension, skip_watermark_area)

        # 计算最小尺寸（考虑整幅或中间60%可用区域），使用RGB 3通道
        side = self._calculate_required_image_size_v2(file_header, bits_per_channel, skip_watermark_area)

        # 生成纯色画布 RGB（中性灰）
        image = np.ones((side, side, 3), dtype=np.uint8) * 128

        # 嵌入数据
        embedded = self._embed_data_multi_bit(image, file_header, bits_per_channel, skip_watermark_area)
        return embedded

    def _calculate_required_image_size_v2(self, data: bytes, bits_per_channel: int, skip_watermark_area: bool) -> int:
        # 数据前置长度字段 4 字节（与V1一致，便于解码）
        bytes_needed = len(data) + 4
        bits_needed = bytes_needed * 8

        # 每像素可用位数：3 通道(RGB) * bits_per_channel
        capacity_per_pixel = 3 * bits_per_channel

        if skip_watermark_area:
            # 连续近似：仅跳过顶部6%，usable ≈ 0.94 * S^2
            side0 = int(np.ceil(np.sqrt(bits_needed / float(capacity_per_pixel * 0.94))))
            side_length = max(64, side0)
            side_length = ((side_length + 3) // 4) * 4
            # 离散校验
            while True:
                top_skip = int(np.floor(side_length * 0.06))
                bottom_skip = 0
                available_height = side_length - top_skip - bottom_skip
                available_pixels = max(0, available_height) * side_length
                available_bits = available_pixels * capacity_per_pixel
                if available_bits >= bits_needed:
                    break
                side_length += 4
        else:
            required_pixels = int(np.ceil(bits_needed / float(capacity_per_pixel)))
            side_length = int(np.ceil(np.sqrt(required_pixels)))
            side_length = max(64, side_length)
            side_length = ((side_length + 3) // 4) * 4
        return side_length

    def _embed_data_multi_bit(self, image: np.ndarray, file_header: bytes, bits_per_channel: int, skip_watermark_area: bool) -> np.ndarray:
        height, width = image.shape[0], image.shape[1]
        embedded = image.copy()

        # 构造完整二进制串：前32位长度 + 数据
        data_length = len(file_header)
        length_binary = format(data_length, '032b')
        data_binary = ''.join(format(byte, '08b') for byte in file_header)
        full_binary = length_binary + data_binary

        channels = embedded.shape[2]
        if skip_watermark_area:
            top_skip = int(np.floor(height * 0.06))
            bottom_skip = 0
            start_row = top_skip
            end_row = height - bottom_skip
            usable_rows = max(0, end_row - start_row)
            total_capacity_bits = usable_rows * width * channels * bits_per_channel
        else:
            start_row = 0
            end_row = height
            total_capacity_bits = height * width * channels * bits_per_channel
        if len(full_binary) > total_capacity_bits:
            # 理论上不会发生，因为尺寸已按容量计算
            raise ValueError("容量计算不足，无法写入全部数据")

        bit_index = 0
        mask = (1 << bits_per_channel) - 1
        clear_mask = 0xFF ^ mask  # 清除最低 bits_per_channel 位

        for i in range(start_row, end_row):
            for j in range(width):
                for c in range(channels):
                    if bit_index >= len(full_binary):
                        break
                    # 取接下来 bits_per_channel 位，不足则右侧用0补齐
                    remaining = len(full_binary) - bit_index
                    take = min(bits_per_channel, remaining)
                    chunk = full_binary[bit_index:bit_index + take]
                    if take < bits_per_channel:
                        chunk = chunk + '0' * (bits_per_channel - take)
                    value = int(chunk, 2) & mask

                    original = embedded[i, j, c]
                    embedded[i, j, c] = (original & clear_mask) | value

                    bit_index += take
                if bit_index >= len(full_binary):
                    break
            if bit_index >= len(full_binary):
                break

        # 若跳过水印区域，将上下留空区域填充为中性灰（提升PNG压缩率）
        if skip_watermark_area:
            if start_row > 0:
                rng = np.random.default_rng()
                noise_top = rng.integers(0, 256, size=(start_row, width, channels), dtype=np.uint8)
                embedded[:start_row, :, :] = noise_top

        return embedded

    def _create_file_header(self, file_data: bytes, file_extension: str, skip_watermark_area: bool) -> bytes:
        """
        V2 文件头（嵌入在32位总长度前缀之后），增加稳健定位：
        [MAGIC(4)='TTv2'][HDR_LEN(4)][CRC16(2)][version(1)][flags(1)][ext_len(1)][ext][data_len(4)][data]
        - version: 2
        - flags: bit0=skip_watermark_area, bit1=multi_file_package
        - CRC16 计算范围：从 version 起，到 data 末尾（不含 MAGIC/HDR_LEN/CRC 本身）
        外层仍保留32位长度前缀，值=上述整个块的字节数，便于快速截断。
        """
        extension_bytes = file_extension.encode('utf-8')
        extension_length = len(extension_bytes)
        if extension_length > 255:
            raise ValueError(f"文件扩展名太长: {file_extension}")
        version = 2
        flags = 0x01 if skip_watermark_area else 0x00
        # 如果是多文件包，设置bit1
        if file_extension == "multi":
            flags |= 0x02

        # 内部头+数据
        inner = bytearray()
        inner.append(version)
        inner.append(flags)
        inner.append(extension_length)
        inner.extend(extension_bytes)
        inner.extend(len(file_data).to_bytes(4, 'big'))
        inner.extend(file_data)

        # CRC16-CCITT (0x1021, init 0xFFFF)
        crc = self._crc16_ccitt(bytes(inner))

        payload = bytearray()
        payload.extend(b'TTv2')                # MAGIC
        payload.extend(len(inner).to_bytes(4, 'big'))  # HDR_LEN (uint32)
        payload.extend(crc.to_bytes(2, 'big'))         # CRC16
        payload.extend(inner)                          # 实际头+数据
        return bytes(payload)

    def _crc16_ccitt(self, data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                if (crc & 0x8000) != 0:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def _create_multi_file_package_data(self, numpy_images, png_compression):
        """
        创建多文件数据包，参考hide_v2.js的encodeMultipleFilesToImageV2逻辑
        返回合并后的数据包字节流
        """
        packets = []
        temp_files = []
        
        print(f"[V2][ENC] 开始创建多文件数据包，图片数量: {len(numpy_images)}")
        
        try:
            for i, img in enumerate(numpy_images):
                print(f"[V2][ENC] 处理图片 {i + 1}/{len(numpy_images)}")
                
                # 为每个图片创建临时PNG文件
                temp_png = self.utils.image_to_png(img, png_compression)
                temp_files.append(temp_png)
                
                # 读取PNG文件数据
                with open(temp_png, 'rb') as f:
                    file_data = f.read()
                
                print(f"[V2][ENC] 图片 {i + 1} PNG文件大小: {len(file_data)} 字节")
                
                # 构建数据包（参考JS版本的buildPacket）
                packet = self._build_packet_v2(2, 0, "png", file_data)
                packets.append(packet)
                
                print(f"[V2][ENC] 图片 {i + 1} 数据包大小: {len(packet)} 字节")
                print(f"[V2][ENC] 图片 {i + 1} 数据包前16字节: {packet[:16].hex()}")
            
            # 将所有数据包合并成一个连续的字节流
            total_length = sum(len(packet) for packet in packets)
            combined_packet = bytearray(total_length)
            
            print(f"[V2][ENC] 开始合并数据包，总长度: {total_length} 字节")
            
            offset = 0
            for i, packet in enumerate(packets):
                print(f"[V2][ENC] 合并数据包 {i + 1}，偏移: {offset}，长度: {len(packet)}")
                combined_packet[offset:offset + len(packet)] = packet
                offset += len(packet)
            
            print(f"[V2][ENC] 合并后数据包前32字节: {bytes(combined_packet[:32]).hex()}")
            
            # 验证每个数据包的Magic标识符
            offset = 0
            for i, packet in enumerate(packets):
                print(f"[V2][ENC] 验证数据包 {i+1}: 偏移 {offset}, Magic: {packet[:4]}")
                offset += len(packet)
            
            return bytes(combined_packet)
            
        finally:
            # 清理临时PNG文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def _create_storage_image_in_memory_v2_multi(self, combined_packet: bytes, bits_per_channel: int, skip_watermark_area: bool) -> np.ndarray:
        """
        为多文件数据包创建存储图片，直接将数据包写入RGB通道
        参考JS版本的writeMultiplePacketsToCanvasRGB逻辑
        
        修改：多文件打包时忽略水印区域，使用整张图片的容量
        """
        print(f"[V2][ENC] 创建多文件存储图片，数据包大小: {len(combined_packet)} 字节")
        
        # 计算最小尺寸，多文件打包时使用整张图片的容量
        # 多文件数据包不需要额外的长度前缀，直接使用数据包大小
        bytes_needed = len(combined_packet)
        bits_needed = bytes_needed * 8
        
        # 每像素可用位数：3 通道(RGB) * bits_per_channel
        capacity_per_pixel = 3 * bits_per_channel
        
        # 修改：多文件打包时忽略水印区域，使用整张图片的容量
        # 无论skip_watermark_area设置如何，多文件打包都使用整张图片
        required_pixels = int(np.ceil(bits_needed / float(capacity_per_pixel)))
        side_length = int(np.ceil(np.sqrt(required_pixels)))
        side_length = max(64, side_length)
        side_length = ((side_length + 3) // 4) * 4
        
        print(f"[V2][ENC] 多文件打包模式：使用整张图片容量，计算所需图片尺寸: {side_length}x{side_length}")

        # 生成纯色画布 RGB（中性灰）
        image = np.ones((side_length, side_length, 3), dtype=np.uint8) * 128
        print(f"[V2][ENC] 创建画布完成，尺寸: {image.shape}")

        # 嵌入数据（多文件数据包不需要额外的文件头）
        print(f"[V2][ENC] 开始嵌入多文件数据包")
        embedded = self._embed_data_multi_bit_direct(image, combined_packet, bits_per_channel, skip_watermark_area)
        print(f"[V2][ENC] 多文件数据包嵌入完成")
        
        return embedded

    def _estimate_first_file_size(self, data_bytes: bytes) -> int:
        """
        估算第一个文件的大小
        用于智能水印控制，确保第一个文件在扫描区域内
        """
        if len(data_bytes) < 4:
            return 0
        
        # 查找第一个Magic标识符
        first_magic_pos = data_bytes.find(b'TTv2')
        if first_magic_pos == -1:
            return 0
        
        # 查找第二个Magic标识符
        second_magic_pos = data_bytes.find(b'TTv2', first_magic_pos + 4)
        if second_magic_pos == -1:
            # 只有一个文件，返回整个数据包大小
            return len(data_bytes)
        
        # 返回第一个文件的大小
        return second_magic_pos

    def _embed_data_multi_bit_direct(self, image: np.ndarray, data_bytes: bytes, bits_per_channel: int, skip_watermark_area: bool) -> np.ndarray:
        """
        直接将字节数据嵌入到图片中（用于多文件数据包）
        参考JS版本的writeMultiplePacketsToCanvasRGB逻辑
        注意：多文件数据包不添加32位长度前缀，直接写入多个独立的数据包
        
        修改：多文件打包时忽略水印区域，直接从0字节开始写入
        """
        height, width = image.shape[0], image.shape[1]
        embedded = image.copy()

        # 多文件数据包：直接写入数据，不添加长度前缀
        data_binary = ''.join(format(byte, '08b') for byte in data_bytes)
        full_binary = data_binary  # 不添加长度前缀
        
        print(f"[V2][ENC] 多文件数据包嵌入: 原始数据 {len(data_bytes)} 字节，二进制长度 {len(full_binary)} 位")
        
        # 验证数据包中的Magic标识符
        if len(data_bytes) >= 4:
            first_magic = data_bytes[:4]
            print(f"[V2][ENC] 数据包第一个Magic: {first_magic}")
            
            # 查找所有Magic标识符
            magic_count = 0
            for i in range(0, len(data_bytes) - 3):
                if data_bytes[i:i+4] == b'TTv2':
                    magic_count += 1
                    print(f"[V2][ENC] 找到Magic标识符 {magic_count} 在偏移 {i}")
            print(f"[V2][ENC] 总共找到 {magic_count} 个Magic标识符")

        channels = embedded.shape[2]
        
        # 修改：多文件打包时忽略水印区域，直接从0字节开始
        # 无论skip_watermark_area设置如何，多文件打包都从顶部开始写入
        start_row = 0
        end_row = height
        total_capacity_bits = height * width * channels * bits_per_channel
        print(f"[V2][ENC] 多文件打包模式：忽略水印区域，从0字节开始写入")
        
        if len(full_binary) > total_capacity_bits:
            # 理论上不会发生，因为尺寸已按容量计算
            raise ValueError("容量计算不足，无法写入全部数据")

        bit_index = 0
        mask = (1 << bits_per_channel) - 1
        clear_mask = 0xFF ^ mask  # 清除最低 bits_per_channel 位

        for i in range(start_row, end_row):
            for j in range(width):
                for c in range(channels):
                    if bit_index >= len(full_binary):
                        break
                    # 取接下来 bits_per_channel 位，不足则右侧用0补齐
                    remaining = len(full_binary) - bit_index
                    take = min(bits_per_channel, remaining)
                    chunk = full_binary[bit_index:bit_index + take]
                    if take < bits_per_channel:
                        chunk = chunk + '0' * (bits_per_channel - take)
                    value = int(chunk, 2) & mask

                    original = embedded[i, j, c]
                    embedded[i, j, c] = (original & clear_mask) | value

                    bit_index += take
                if bit_index >= len(full_binary):
                    break
            if bit_index >= len(full_binary):
                break

        # 多文件打包时不需要填充随机噪声，因为数据从顶部开始写入
        # 保持原有的中性灰背景即可
        print(f"[V2][ENC] 多文件打包模式：数据已从顶部开始写入，无需填充噪声")

        return embedded

    def _build_packet_v2(self, version, flags, ext, data_bytes):
        """
        构建V2数据包，参考JS版本的buildPacket函数
        [MAGIC(4)='TTv2'][HDR_LEN(4)][CRC16(2)][version(1)][flags(1)][ext_len(1)][ext][data_len(4)][data]
        """
        extension_bytes = ext.encode('utf-8') if ext else b''
        if len(extension_bytes) > 255:
            raise ValueError('扩展名过长')
        
        print(f"[V2][ENC] 构建数据包: version={version}, flags={flags}, ext='{ext}', data_len={len(data_bytes)}")
        
        # 内部数据
        inner = bytearray()
        inner.append(version)
        inner.append(flags)
        inner.append(len(extension_bytes))
        inner.extend(extension_bytes)
        inner.extend(len(data_bytes).to_bytes(4, 'big'))
        inner.extend(data_bytes)
        
        print(f"[V2][ENC] 内部数据长度: {len(inner)} 字节")
        
        # CRC16-CCITT
        crc = self._crc16_ccitt(bytes(inner))
        print(f"[V2][ENC] CRC16: {crc:04x}")
        
        # 完整数据包
        packet = bytearray()
        packet.extend(b'TTv2')  # MAGIC
        packet.extend(len(inner).to_bytes(4, 'big'))  # HDR_LEN
        packet.extend(crc.to_bytes(2, 'big'))  # CRC16
        packet.extend(inner)  # 内部数据
        
        print(f"[V2][ENC] 完整数据包长度: {len(packet)} 字节")
        print(f"[V2][ENC] 数据包Magic: {packet[:4]}")
        
        return bytes(packet)


# 节点类定义完成


