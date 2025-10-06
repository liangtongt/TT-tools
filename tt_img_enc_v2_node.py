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
                "compress_level": ("INT", {"default": 6, "min": 0, "max": 9}),
                # 跳过上下各20%区域（避免水印/边界干扰），仅使用中间60%写入
                "skip_watermark_area": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "usage_notes": ("STRING", {"default": "V2：自动使用每通道8位（最大容量），可选跳过上下20%（中间60%）水印安全区。教程：https://b23.tv/RbvaMeW\nB站：我是小斯呀", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True

    def process_images(self, images, fps=16.0, compress_level=6, skip_watermark_area=True, audio=None, usage_notes=None):
        """
        将输入图片打包为文件（单图->PNG，多图->MP4[可选音频]），并以更高位宽的隐写方式写入到存储图片中。
        与V1不同：
        - 使用整张图片区域（不保留水印安全边界）
        - 使用可配置的每通道位数（默认2位）提升容量
        """
        try:
            # 自动使用最大位宽
            bits_per_channel = 8

            num_images = len(images)
            numpy_images = self._batch_convert_images(images)

            temp_file = None
            file_extension = None

            if num_images > 1:
                if audio is not None:
                    temp_file = self.utils.images_to_mp4_with_audio(numpy_images, fps, audio)
                else:
                    temp_file = self.utils.images_to_mp4(numpy_images, fps)
                file_extension = "mp4"
            else:
                temp_file = self.utils.image_to_png(numpy_images[0], compress_level)
                file_extension = "png"

            output_image = self._create_storage_image_in_memory_v2(temp_file, file_extension, bits_per_channel, skip_watermark_area)

            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            output_tensor = self._convert_to_tensor(output_image)

            if usage_notes:
                print(f"=== TT img enc V2 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"输出图片尺寸: {output_image.shape[1]}x{output_image.shape[0]}")
                print(f"文件类型: {file_extension}")
                print(f"每通道位数: {bits_per_channel}")
                print(f"跳过上下20%: {bool(skip_watermark_area)}")

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

        # 计算最小尺寸（考虑整幅或中间60%可用区域）
        side = self._calculate_required_image_size_v2(file_header, bits_per_channel, skip_watermark_area)

        # 生成纯色画布（中性灰），便于多位隐写而不过度可见
        image = np.ones((side, side, 3), dtype=np.uint8) * 128

        # 嵌入数据
        embedded = self._embed_data_multi_bit(image, file_header, bits_per_channel, skip_watermark_area)
        return embedded

    def _calculate_required_image_size_v2(self, data: bytes, bits_per_channel: int, skip_watermark_area: bool) -> int:
        # 数据前置长度字段 4 字节（与V1一致，便于解码）
        bytes_needed = len(data) + 4
        bits_needed = bytes_needed * 8

        # 每像素可用位数：3 通道 * bits_per_channel
        capacity_per_pixel = 3 * bits_per_channel

        if skip_watermark_area:
            # 连续近似：usable ≈ 0.6 * S^2
            side0 = int(np.ceil(np.sqrt(bits_needed / float(capacity_per_pixel * 0.6))))
            side_length = max(64, side0)
            side_length = ((side_length + 3) // 4) * 4
            # 离散校验
            while True:
                top_skip = int(np.floor(side_length * 0.20))
                bottom_skip = int(np.floor(side_length * 0.20))
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

        if skip_watermark_area:
            top_skip = int(np.floor(height * 0.20))
            bottom_skip = int(np.floor(height * 0.20))
            start_row = top_skip
            end_row = height - bottom_skip
            usable_rows = max(0, end_row - start_row)
            total_capacity_bits = usable_rows * width * 3 * bits_per_channel
        else:
            start_row = 0
            end_row = height
            total_capacity_bits = height * width * 3 * bits_per_channel
        if len(full_binary) > total_capacity_bits:
            # 理论上不会发生，因为尺寸已按容量计算
            raise ValueError("容量计算不足，无法写入全部数据")

        bit_index = 0
        mask = (1 << bits_per_channel) - 1
        clear_mask = 0xFF ^ mask  # 清除最低 bits_per_channel 位

        for i in range(start_row, end_row):
            for j in range(width):
                for c in range(3):
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

        # 若跳过水印区域，为上下留空区域添加随机噪点，仅扰动低 bits_per_channel 位，以保持整体观感一致
        if skip_watermark_area:
            rng = np.random.default_rng()
            # 顶部区域 [0, start_row)
            if start_row > 0:
                # 生成随机低位值，形状: (start_row, width, 3)
                noise_top = rng.integers(0, mask + 1, size=(start_row, width, 3), dtype=np.uint8)
                # 应用到低位
                embedded[:start_row, :, :] = (embedded[:start_row, :, :] & clear_mask) | noise_top
            # 底部区域 [end_row, height)
            if end_row < height:
                rows_bottom = height - end_row
                noise_bottom = rng.integers(0, mask + 1, size=(rows_bottom, width, 3), dtype=np.uint8)
                embedded[end_row:, :, :] = (embedded[end_row:, :, :] & clear_mask) | noise_bottom

        return embedded

    def _create_file_header(self, file_data: bytes, file_extension: str, skip_watermark_area: bool) -> bytes:
        """
        V2 文件头（嵌入在32位长度之后）：
        [version:1][flags:1][ext_len:1][ext:ext_len][data_len:4][data]
        - version: 2
        - flags: bit0=skip_watermark_area, 其他保留为0
        保持整体仍以32位长度前缀封装，便于与既有读取策略一致。
        """
        extension_bytes = file_extension.encode('utf-8')
        extension_length = len(extension_bytes)
        if extension_length > 255:
            raise ValueError(f"文件扩展名太长: {file_extension}")
        version = 2
        flags = 0x01 if skip_watermark_area else 0x00
        header = bytearray()
        header.append(version)
        header.append(flags)
        header.append(extension_length)
        header.extend(extension_bytes)
        header.extend(len(file_data).to_bytes(4, 'big'))
        header.extend(file_data)
        return bytes(header)


# 节点类定义完成


