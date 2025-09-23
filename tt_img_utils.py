import os
from PIL import Image
import numpy as np
import cv2
from typing import List
import subprocess
import tempfile
import uuid


class TTImgUtils:
    """TT图片处理工具类，包含编码节点的共同方法"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        # 创建必要的目录
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def images_to_mp4(self, images: List[np.ndarray], fps: float) -> str:
        """将多张图片转换为MP4视频（使用FFmpeg）"""
        random_suffix = str(uuid.uuid4())[:8]
        temp_path = os.path.join(self.temp_dir, f"temp_video_{random_suffix}.mp4")
        
        # 获取图片尺寸
        height, width = images[0].shape[:2]
        
        # 确保尺寸是偶数（H.264要求）
        if width % 2 != 0:
            width += 1
        if height % 2 != 0:
            height += 1
        
        # 优化：批量处理图片尺寸调整
        resized_images = self._batch_resize_images(images, width, height)
        
        # 直接使用FFmpeg创建视频
        return self._create_video_with_ffmpeg(resized_images, temp_path, fps, width, height)
    
    def _batch_resize_images(self, images: List[np.ndarray], target_width: int, target_height: int) -> List[np.ndarray]:
        """批量调整图片尺寸，优化性能"""
        resized_images = []
        
        for img in images:
            if img.shape[1] != target_width or img.shape[0] != target_height:
                # 使用更高效的插值方法
                img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img
            resized_images.append(img_resized)
        
        return resized_images
    
    def _create_video_with_ffmpeg(self, images: List[np.ndarray], output_path: str, fps: float, width: int, height: int) -> str:
        """使用FFmpeg直接创建视频（备用方法）"""
        try:
            # 创建临时目录存储图片
            random_suffix = str(uuid.uuid4())[:8]
            temp_dir = os.path.join(self.temp_dir, f"ffmpeg_temp_{random_suffix}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有图片为临时文件
            temp_images = []
            for i, img in enumerate(images):
                temp_img_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                
                # 确保图片是RGB格式（FFmpeg期望RGB，不是BGR）
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # 如果是RGB格式，转换为BGR供cv2.imwrite使用
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    # 如果是RGBA格式，先转RGB再转BGR
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                else:
                    # 灰度图转BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                cv2.imwrite(temp_img_path, img_bgr)
                temp_images.append(temp_img_path)
            
            # 使用FFmpeg创建视频（统一配置，添加颜色空间和优化参数）
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264',  # 使用H.264编码器
                '-pix_fmt', 'yuv420p',  # iPhone兼容的像素格式
                '-crf', '19',  # 高质量压缩
                '-vf', 'scale=out_color_matrix=bt709',  # 视频滤镜：颜色矩阵转换
                '-color_range', 'tv',  # 颜色范围
                '-colorspace', 'bt709',  # 颜色空间
                '-color_primaries', 'bt709',  # 颜色原色
                '-color_trc', 'bt709',  # 颜色传输特性
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 清理临时文件
            for temp_img in temp_images:
                if os.path.exists(temp_img):
                    os.remove(temp_img)
            os.rmdir(temp_dir)
            
            if result.returncode == 0:
                print("成功使用FFmpeg创建视频")
                return output_path
            else:
                print(f"FFmpeg创建视频失败: {result.stderr}")
                raise RuntimeError(f"FFmpeg错误: {result.stderr}")
                
        except FileNotFoundError:
            print("FFmpeg不可用")
            raise RuntimeError("无法创建视频：OpenCV和FFmpeg都不可用")
        except Exception as e:
            print(f"FFmpeg创建视频异常: {e}")
            raise RuntimeError(f"视频创建失败: {e}")
    
    def images_to_mp4_with_audio(self, images: List[np.ndarray], fps: float, audio) -> str:
        """将多张图片转换为带音频的MP4视频（iPhone兼容版本）"""
        # 先生成无音频的MP4
        video_path = self.images_to_mp4(images, fps)
        
        # 如果没有音频，直接返回视频
        if audio is None:
            return video_path
        
        # 创建带音频的MP4
        random_suffix = str(uuid.uuid4())[:8]
        audio_video_path = os.path.join(self.temp_dir, f"temp_video_with_audio_{random_suffix}.mp4")
        
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
        random_suffix = str(uuid.uuid4())[:8]
        audio_path = os.path.join(self.temp_dir, f"temp_audio_{random_suffix}.wav")
        
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
        
        # 如果audio是LazyAudioMap（ComfyUI-VideoHelperSuite格式）
        elif hasattr(audio, '__class__') and 'LazyAudioMap' in str(type(audio)):
            try:
                print(f"处理LazyAudioMap: {type(audio)}")
                
                # 根据调试信息，LazyAudioMap有'file'属性
                if hasattr(audio, 'file') and audio.file:
                    print(f"LazyAudioMap.file: {audio.file}")
                    if os.path.exists(audio.file):
                        import shutil
                        shutil.copy2(audio.file, audio_path)
                        print(f"成功复制音频文件: {audio.file} -> {audio_path}")
                        return audio_path
                    else:
                        print(f"音频文件不存在: {audio.file}")
                
                # 尝试通过索引访问音频数据
                try:
                    # LazyAudioMap可能支持索引访问
                    audio_data = audio[0] if len(audio) > 0 else None
                    if audio_data is not None:
                        if isinstance(audio_data, np.ndarray):
                            import soundfile as sf
                            sf.write(audio_path, audio_data, 44100)
                            print(f"成功从索引获取音频数据")
                            return audio_path
                        elif hasattr(audio_data, 'cpu'):
                            audio_np = audio_data.cpu().numpy()
                            import soundfile as sf
                            sf.write(audio_path, audio_np, 44100)
                            print(f"成功从索引获取音频张量数据")
                            return audio_path
                except Exception as e:
                    print(f"通过索引访问音频数据失败: {e}")
                
                # 尝试通过get方法获取音频数据
                try:
                    audio_data = audio.get('samples') or audio.get('data') or audio.get('audio')
                    if audio_data is not None:
                        if isinstance(audio_data, np.ndarray):
                            import soundfile as sf
                            sf.write(audio_path, audio_data, 44100)
                            print(f"成功通过get方法获取音频数据")
                            return audio_path
                        elif hasattr(audio_data, 'cpu'):
                            audio_np = audio_data.cpu().numpy()
                            import soundfile as sf
                            sf.write(audio_path, audio_np, 44100)
                            print(f"成功通过get方法获取音频张量数据")
                            return audio_path
                except Exception as e:
                    print(f"通过get方法获取音频数据失败: {e}")
                
                # 尝试迭代访问音频数据
                try:
                    for item in audio:
                        if isinstance(item, np.ndarray):
                            import soundfile as sf
                            sf.write(audio_path, item, 44100)
                            print(f"成功通过迭代获取音频数据")
                            return audio_path
                        elif hasattr(item, 'cpu'):
                            audio_np = item.cpu().numpy()
                            import soundfile as sf
                            sf.write(audio_path, audio_np, 44100)
                            print(f"成功通过迭代获取音频张量数据")
                            return audio_path
                        break  # 只取第一个元素
                except Exception as e:
                    print(f"通过迭代获取音频数据失败: {e}")
                    
            except Exception as e:
                print(f"处理LazyAudioMap失败: {e}")
        
        # 尝试通过属性访问获取音频数据
        try:
            # 检查是否有常见的音频属性
            audio_attrs = ['audio', 'data', 'waveform', 'signal']
            for attr in audio_attrs:
                if hasattr(audio, attr):
                    audio_data = getattr(audio, attr)
                    if isinstance(audio_data, np.ndarray):
                        import soundfile as sf
                        sf.write(audio_path, audio_data, 44100)
                        return audio_path
                    elif hasattr(audio_data, 'cpu'):
                        audio_np = audio_data.cpu().numpy()
                        import soundfile as sf
                        sf.write(audio_path, audio_np, 44100)
                        return audio_path
        except Exception as e:
            print(f"通过属性访问音频数据失败: {e}")
        
        # 尝试将对象转换为字符串路径
        try:
            audio_str = str(audio)
            if os.path.exists(audio_str):
                import shutil
                shutil.copy2(audio_str, audio_path)
                return audio_path
        except Exception as e:
            print(f"尝试字符串路径失败: {e}")
        
        raise ValueError(f"不支持的音频格式: {type(audio)}")
    
    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str):
        """使用FFmpeg合并音频和视频"""
        try:
            # 使用FFmpeg命令合并音频和视频
            cmd = [
                'ffmpeg',
                '-i', video_path,  # 输入视频
                '-i', audio_path,  # 输入音频
                '-map', '0:v:0',   # 映射第一个输入的视频流
                '-map', '1:a:0',   # 映射第二个输入的音频流
                '-c:v', 'copy',    # 视频编码器：直接复制
                '-c:a', 'aac',     # 音频编码器：AAC
                '-shortest',       # 以最短的流为准（视频长度为准）
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
    
    def image_to_png(self, image: np.ndarray, compress_level: int = 6) -> str:
        """将单张图片转换为PNG格式（完全按照ComfyUI标准转换算法）"""
        random_suffix = str(uuid.uuid4())[:8]
        temp_path = os.path.join(self.temp_dir, f"temp_image_{random_suffix}.png")
        
        # 完全按照ComfyUI标准转换算法
        # 确保图片是numpy数组
        if hasattr(image, 'cpu'):
            # 如果是torch张量，转换为numpy
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        # 检查数据范围并正确转换
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            # 如果是浮点数，检查范围
            if image_np.max() <= 1.0:
                # 范围是0-1，需要乘以255
                i = 255. * image_np
            else:
                # 范围已经是0-255，直接使用
                i = image_np
        else:
            # 整数类型，假设已经是0-255范围
            i = image_np.astype(np.float32)
        
        # 使用ComfyUI标准的clip和转换
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        
        # 按照ComfyUI标准，使用PIL Image保存
        img = Image.fromarray(img_array)
        img.save(temp_path, compress_level=compress_level, optimize=True)
        
        return temp_path
    
    def calculate_required_image_size(self, data: bytes) -> int:
        """计算存储数据所需的图片尺寸（优化存储效率，考虑水印区域）"""
        # 每个像素3个通道，每个通道1位
        # 预留嵌入阶段的长度标记开销（前32位 = 4字节）
        bytes_needed = len(data) + 4
        bits_needed = bytes_needed * 8
        required_pixels = int(np.ceil(bits_needed / 3.0))

        # 仅在中间60%高度写入数据（上下各预留20%）
        # 先用连续模型近似：usable_pixels ≈ 0.6 * S^2
        # 得 S0 ≈ sqrt(required_pixels / 0.6)
        s0 = int(np.ceil(np.sqrt(required_pixels / 0.6)))
        side_length = max(64, s0)
        # 4的倍数对齐
        side_length = ((side_length + 3) // 4) * 4
        
        # 使用离散模型做严格校验：usable_pixels = (S - floor(0.2S) - floor(0.2S)) * S
        # 若不足则按步进增长直至满足
        while True:
            top_skip = int(np.floor(side_length * 0.20))
            bottom_skip = int(np.floor(side_length * 0.20))
            available_height = side_length - top_skip - bottom_skip
            available_pixels = max(0, available_height) * side_length
            if available_pixels >= required_pixels:
                break
            side_length += 4  # 维持4对齐递增
        
        return side_length
    
    def embed_file_data_in_image(self, image: np.ndarray, file_header: bytes) -> np.ndarray:
        """将文件数据直接嵌入到图片中（不使用ZIP压缩）"""
        # 仅在中间60%高度写入数据（上下各预留20%，用于水印/安全边界）
        height, width = image.shape[0], image.shape[1]
        top_skip = int(height * 0.20)
        bottom_skip = int(height * 0.20)
        start_row = top_skip
        end_row = height - bottom_skip  # 不包含该行
        available_height = max(0, end_row - start_row)
        
        # 检查数据大小是否超过可用图片容量；若不足则自动扩展画布并重嵌入
        max_data_size = available_height * width * 3 // 8  # 每个像素3通道，每8位1字节
        if len(file_header) > max_data_size:
            required_size = self.calculate_required_image_size(file_header)
            print(
                f"容量不足：数据 {len(file_header)}B > 可用 {max_data_size}B，"
                f"自动扩展到 {required_size}x{required_size} 重新嵌入"
            )
            larger_image = self.create_storage_image(required_size)
            return self.embed_file_data_in_image(larger_image, file_header)
        
        print(
            f"嵌入文件数据: {len(file_header)} 字节到 {height}x{width} 图片（写入行: {start_row}~{end_row-1}，"
            f"顶部预留≈20%，底部预留≈20%）"
        )
        
        # 复制图片
        embedded_image = image.copy()
        
        # 将文件数据转换为二进制字符串
        data_binary = ''.join(format(byte, '08b') for byte in file_header)
        
        # 添加数据长度标记（前32位）
        data_length = len(file_header)
        length_binary = format(data_length, '032b')
        full_binary = length_binary + data_binary
        
        # 嵌入数据到图片的LSB（仅在中间可用区域写入）
        data_index = 0
        for i in range(start_row, end_row):
            for j in range(width):
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
    
    def create_storage_image_with_file_data(self, file_header: bytes) -> np.ndarray:
        """直接在内存中创建存储图片并嵌入文件数据（优化版本）"""
        # 计算所需的图片尺寸
        required_size = self.calculate_required_image_size(file_header)
        
        # 创建纯色存储图片（最小尺寸，最大存储效率）
        storage_image = self.create_storage_image(required_size)
        
        # 将文件数据直接嵌入到图片中
        embedded_image = self.embed_file_data_in_image(storage_image, file_header)
        
        return embedded_image
