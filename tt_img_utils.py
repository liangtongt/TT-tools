import os
from PIL import Image
import numpy as np
import cv2
from typing import List
import subprocess
import tempfile


class TTImgUtils:
    """TT图片处理工具类，包含编码节点的共同方法"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        # 创建必要的目录
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def images_to_mp4(self, images: List[np.ndarray], fps: float) -> str:
        """将多张图片转换为MP4视频（iPhone兼容版本）"""
        temp_path = os.path.join(self.temp_dir, "temp_video.mp4")
        
        # 获取图片尺寸
        height, width = images[0].shape[:2]
        
        # 确保尺寸是偶数（H.264要求）
        if width % 2 != 0:
            width += 1
        if height % 2 != 0:
            height += 1
        
        # 调整图片尺寸
        resized_images = []
        for img in images:
            if img.shape[1] != width or img.shape[0] != height:
                img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img
            resized_images.append(img_resized)
        
        # 优先使用H.264编码器，iPhone最佳兼容性
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # 如果H264不可用，尝试其他编码器
            print("H264编码器不可用，尝试其他编码器...")
            codecs_to_try = [
                ('XVID', 'XVID编码器'),
                ('MJPG', 'Motion JPEG编码器'),
                ('mp4v', 'MP4V编码器')
            ]
            
            for codec, description in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"使用编码器: {description}")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    print(f"编码器 {codec} 不可用: {e}")
                    if out:
                        out.release()
                        out = None
        
        if out is None or not out.isOpened():
            raise RuntimeError("无法创建视频写入器，请检查OpenCV安装")
        
        try:
            for img in resized_images:
                # 确保图片是BGR格式（OpenCV要求）
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # RGB转BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    # 灰度图转BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # 确保图片是uint8类型
                if img_bgr.dtype != np.uint8:
                    img_bgr = (img_bgr * 255).astype(np.uint8)
                
                # 确保图片数据范围在0-255之间
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
                
                out.write(img_bgr)
        finally:
            out.release()
        
        return temp_path
    
    def images_to_mp4_with_audio(self, images: List[np.ndarray], fps: float, audio) -> str:
        """将多张图片转换为带音频的MP4视频（iPhone兼容版本）"""
        # 先生成无音频的MP4
        video_path = self.images_to_mp4(images, fps)
        
        # 如果没有音频，直接返回视频
        if audio is None:
            return video_path
        
        # 创建带音频的MP4
        audio_video_path = os.path.join(self.temp_dir, "temp_video_with_audio.mp4")
        
        try:
            # 处理音频数据
            audio_path = self._process_audio_input(audio)
            
            # 使用FFmpeg合成音频和视频
            self._merge_audio_video(video_path, audio_path, audio_video_path)
            
            # 清理临时音频文件
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
            # 清理原始视频文件
            if os.path.exists(video_path):
                os.remove(video_path)
            
            return audio_video_path
            
        except Exception as e:
            print(f"音频合成失败: {e}")
            # 如果音频合成失败，返回原始视频
            return video_path
    
    def _process_audio_input(self, audio) -> str:
        """处理音频输入，返回临时音频文件路径"""
        audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        # 如果audio是文件路径
        if isinstance(audio, str) and os.path.exists(audio):
            # 如果是音频文件，直接复制
            import shutil
            shutil.copy2(audio, audio_path)
            return audio_path
        
        # 如果audio是numpy数组（音频数据）
        elif isinstance(audio, np.ndarray):
            # 将numpy数组保存为WAV文件
            import soundfile as sf
            sf.write(audio_path, audio, 44100)  # 默认采样率44.1kHz
            return audio_path
        
        # 如果audio是torch张量
        elif hasattr(audio, 'cpu'):
            # 转换为numpy数组
            audio_np = audio.cpu().numpy()
            import soundfile as sf
            sf.write(audio_path, audio_np, 44100)
            return audio_path
        
        # 如果audio是字典（ComfyUI音频格式）
        elif isinstance(audio, dict):
            if 'samples' in audio:
                # 提取音频数据
                audio_data = audio['samples']
                if hasattr(audio_data, 'cpu'):
                    audio_data = audio_data.cpu().numpy()
                
                # 获取采样率
                sample_rate = audio.get('sample_rate', 44100)
                
                # 保存音频文件
                import soundfile as sf
                sf.write(audio_path, audio_data, sample_rate)
                return audio_path
        
        raise ValueError(f"不支持的音频格式: {type(audio)}")
    
    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str):
        """使用FFmpeg合并音频和视频"""
        try:
            # 使用FFmpeg命令合并音频和视频
            cmd = [
                'ffmpeg',
                '-i', video_path,  # 输入视频
                '-i', audio_path,  # 输入音频
                '-c:v', 'copy',    # 视频编码器：直接复制
                '-c:a', 'aac',     # 音频编码器：AAC
                '-shortest',       # 以最短的流为准
                '-y',              # 覆盖输出文件
                output_path
            ]
            
            # 执行FFmpeg命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg错误: {result.stderr}")
                
        except FileNotFoundError:
            # 如果FFmpeg不可用，尝试使用OpenCV（但OpenCV不支持音频）
            print("FFmpeg不可用，无法添加音频，返回无音频视频")
            import shutil
            shutil.copy2(video_path, output_path)
        except Exception as e:
            print(f"音频合并失败: {e}")
            # 如果合并失败，复制原始视频
            import shutil
            shutil.copy2(video_path, output_path)
    
    def image_to_jpg(self, image: np.ndarray, quality: int = 95) -> str:
        """将单张图片转换为JPG格式（手机兼容版本，无尺寸限制）"""
        temp_path = os.path.join(self.temp_dir, "temp_image.jpg")
        
        # 确保图片数据类型正确
        if image.dtype != np.uint8:
            # 如果是浮点数，假设范围是0-1
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 确保像素值在有效范围内
        image = np.clip(image, 0, 255)
        
        # 转换为PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image, 'RGB')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA转RGB
            pil_image = Image.fromarray(image, 'RGBA').convert('RGB')
        else:
            # 灰度图转RGB
            pil_image = Image.fromarray(image, 'L').convert('RGB')
        
        # 保存为JPG，使用兼容性参数（无尺寸限制）
        pil_image.save(temp_path, 'JPEG', 
                      quality=quality,
                      optimize=True,
                      progressive=True,
                      subsampling=0)  # 禁用子采样，提高兼容性
        
        return temp_path
    
    def image_to_mobile_jpg(self, image: np.ndarray, quality: int = 90) -> str:
        """专门为手机优化的JPG转换（最大兼容性）"""
        temp_path = os.path.join(self.temp_dir, "mobile_image.jpg")
        
        # 确保图片数据类型正确
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # 浮点数范围检查
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 严格限制像素值范围
        image = np.clip(image, 0, 255)
        
        # 处理不同格式的图片
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB图片
            pil_image = Image.fromarray(image, 'RGB')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA图片，转换为RGB
            pil_image = Image.fromarray(image, 'RGBA').convert('RGB')
        elif len(image.shape) == 2:
            # 灰度图片，转换为RGB
            pil_image = Image.fromarray(image, 'L').convert('RGB')
        else:
            raise ValueError(f"不支持的图片格式: {image.shape}")
        
        # 确保图片尺寸合理（避免过大导致内存问题）
        max_size = 4096
        if pil_image.width > max_size or pil_image.height > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 保存为JPG，使用最兼容的参数
        pil_image.save(temp_path, 'JPEG', 
                      quality=quality,
                      optimize=True,
                      progressive=True,
                      subsampling=0,  # 禁用色度子采样
                      qtables='web_low')  # 使用web优化量化表
        
        return temp_path
    
    def image_to_ultra_compatible_jpg(self, image: np.ndarray) -> str:
        """超兼容JPG转换（适用于所有设备）"""
        temp_path = os.path.join(self.temp_dir, "ultra_compatible.jpg")
        
        # 数据类型转换
        if image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 像素值范围限制
        image = np.clip(image, 0, 255)
        
        # 转换为PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image, 'RGB')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            pil_image = Image.fromarray(image, 'RGBA').convert('RGB')
        else:
            pil_image = Image.fromarray(image, 'L').convert('RGB')
        
        # 限制最大尺寸（避免某些设备无法处理大图）
        max_dimension = 2048
        if pil_image.width > max_dimension or pil_image.height > max_dimension:
            pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # 使用最保守的JPG参数
        pil_image.save(temp_path, 'JPEG', 
                      quality=85,  # 适中的质量
                      optimize=True,
                      progressive=False,  # 不使用渐进式，提高兼容性
                      subsampling=0)  # 禁用子采样
        
        return temp_path
    
    def calculate_required_image_size(self, data: bytes) -> int:
        """计算存储数据所需的图片尺寸（优化存储效率，考虑水印区域）"""
        # 每个像素3个通道，每个通道1位
        bits_needed = len(data) * 8
        pixels_needed = bits_needed // 3
        
        # 考虑水印区域（使用5%的比例，与解码节点保持一致）
        # 先估算一个合理的初始尺寸
        estimated_size = int(np.ceil(np.sqrt(pixels_needed * 1.2)))  # 预留20%空间
        estimated_size = max(64, estimated_size)
        estimated_size = ((estimated_size + 3) // 4) * 4  # 4的倍数对齐
        
        # 使用估算尺寸计算水印区域
        watermark_height = int(estimated_size * 0.05)
        
        # 计算需要的总像素数（包括水印区域）
        # 我们需要确保有足够的像素来存储数据，同时为水印区域预留空间
        total_pixels_needed = pixels_needed + (watermark_height * estimated_size)
        
        # 计算正方形图片的边长
        side_length = int(np.ceil(np.sqrt(total_pixels_needed)))
        
        # 确保最小尺寸为64
        side_length = max(64, side_length)
        
        # 向上取整到最近的4的倍数（更小的对齐，提高效率）
        side_length = ((side_length + 3) // 4) * 4
        
        # 验证计算是否正确
        available_pixels = (side_length - watermark_height) * side_length
        required_pixels = pixels_needed
        
        if available_pixels < required_pixels:
            # 如果可用像素不够，增加图片尺寸
            side_length = int(np.ceil(np.sqrt(required_pixels + watermark_height * side_length)))
            side_length = ((side_length + 3) // 4) * 4
        
        return side_length
    
    def embed_file_data_in_image(self, image: np.ndarray, file_header: bytes) -> np.ndarray:
        """将文件数据直接嵌入到图片中（不使用ZIP压缩）"""
        # 计算可用区域（使用5%的比例，与解码节点保持一致）
        watermark_height = int(image.shape[0] * 0.05)
        available_height = image.shape[0] - watermark_height
        
        # 检查数据大小是否超过可用图片容量
        max_data_size = available_height * image.shape[1] * 3 // 8  # 每个像素3通道，每8位1字节
        if len(file_header) > max_data_size:
            raise ValueError(f"文件太大 ({len(file_header)} 字节)，当前图片最大支持 {max_data_size} 字节（已排除水印区域）")
        
        print(f"嵌入文件数据: {len(file_header)} 字节到 {image.shape[0]}x{image.shape[1]} 图片（从第{watermark_height+1}行开始，水印区域高度: {watermark_height}像素）")
        
        # 复制图片
        embedded_image = image.copy()
        
        # 将文件数据转换为二进制字符串
        data_binary = ''.join(format(byte, '08b') for byte in file_header)
        
        # 添加数据长度标记（前32位）
        data_length = len(file_header)
        length_binary = format(data_length, '032b')
        full_binary = length_binary + data_binary
        
        # 嵌入数据到图片的LSB（从水印区域后开始，避开水印区域）
        data_index = 0
        for i in range(watermark_height, image.shape[0]):  # 从水印区域后开始
            for j in range(image.shape[1]):
                for k in range(3):  # RGB通道
                    if data_index < len(full_binary):
                        # 修改最低位
                        if full_binary[data_index] == '1':
                            embedded_image[i, j, k] |= 1
                        else:
                            embedded_image[i, j, k] &= 0xFE
                        data_index += 1
                    else:
                        break
                if data_index >= len(full_binary):
                    break
            if data_index >= len(full_binary):
                break
        
        return embedded_image
    
    def create_storage_image(self, size: int) -> np.ndarray:
        """创建存储图片（纯色背景，最大存储效率）"""
        # 创建纯色背景（灰色，便于LSB隐写）
        image = np.ones((size, size, 3), dtype=np.uint8) * 128
        
        return image
    
    def create_error_image(self, size: int = 512, error_type: str = "enc") -> np.ndarray:
        """创建错误提示图片"""
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # 根据图片尺寸调整文字大小和位置
        scale = size / 512.0
        font_scale = max(0.5, scale)
        thickness = max(1, int(scale))
        
        # 计算文字位置
        text1_pos = (int(50 * scale), int(200 * scale))
        text2_pos = (int(50 * scale), int(250 * scale))
        
        # 根据错误类型设置不同的错误文字
        if error_type == "enc_pw":
            error_text = "Error in TT img enc pw"
        else:
            error_text = "Error in TT img enc"
        
        # 添加错误文字
        cv2.putText(image, error_text, text1_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(image, "Check console for details", text2_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 0, 255), thickness)
        
        return image
    
    def create_storage_image_with_file(self, file_path: str, file_extension: str, file_header: bytes) -> np.ndarray:
        """创建存储图片并嵌入文件（通用版本）"""
        # 计算所需的图片尺寸
        required_size = self.calculate_required_image_size(file_header)
        print(f"文件大小: {len(file_header)} 字节，需要图片尺寸: {required_size}x{required_size}")
        
        # 创建纯色存储图片（最小尺寸，最大存储效率）
        storage_image = self.create_storage_image(required_size)
        
        # 将文件数据直接嵌入到图片中
        embedded_image = self.embed_file_data_in_image(storage_image, file_header)
        
        return embedded_image
