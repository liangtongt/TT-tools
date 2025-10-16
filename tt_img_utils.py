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
    
    def images_to_mp4(self, images: List[np.ndarray], fps: float, crf: int = 19, audio=None) -> str:
        """将多张图片转换为MP4视频（使用FFmpeg），支持可选音频"""
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
        
        # 重新启用管道方法进行测试（已修复MP4兼容性问题）
        print("[PIPE_TEST] 重新启用管道方法进行测试")
        
        if audio is not None:
            # 有音频：一步完成音视频合成
            try:
                return self._create_video_with_audio_pipe(resized_images, temp_path, fps, width, height, crf, audio)
            except Exception as e:
                print(f"[FALLBACK] 音视频管道方法失败，回退到分离方法: {e}")
                return self.images_to_mp4_with_audio(resized_images, fps, audio, crf)
        else:
            # 无音频：只合成视频
            try:
                return self._create_video_with_ffmpeg_pipe(resized_images, temp_path, fps, width, height, crf)
            except Exception as e:
                print(f"[FALLBACK] 视频管道方法失败，回退到文件方法: {e}")
                return self._create_video_with_ffmpeg(resized_images, temp_path, fps, width, height, crf)
    
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
    
    def _create_video_with_ffmpeg_pipe(self, images: List[np.ndarray], output_path: str, fps: float, width: int, height: int, crf: int = 19) -> str:
        """使用FFmpeg管道创建视频（类似ComfyUI-VideoHelperSuite，避免中间文件）"""
        try:
            print(f"[PIPE] 开始使用FFmpeg管道创建视频，帧数: {len(images)}, 尺寸: {width}x{height}, FPS: {fps}, CRF: {crf}")
            
            # 准备图像数据
            frame_data = b''
            for i, img in enumerate(images):
                # 确保图像是RGB格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = img
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    # RGBA转RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:
                    # 灰度图转RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # 确保尺寸正确
                if img_rgb.shape[1] != width or img_rgb.shape[0] != height:
                    img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # 确保数据类型正确
                if img_rgb.dtype != np.uint8:
                    img_rgb = img_rgb.astype(np.uint8)
                
                # 转换为字节数据
                frame_bytes = img_rgb.tobytes()
                frame_data += frame_bytes
                
                if i % 10 == 0:  # 每10帧打印一次进度
                    print(f"[PIPE] 已处理 {i+1}/{len(images)} 帧")
            
            print(f"[PIPE] 图像数据准备完成，总大小: {len(frame_data)} 字节")
            
            # 使用FFmpeg管道（修复MP4兼容性问题）
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',           # 输入格式：原始视频
                '-pix_fmt', 'rgb24',        # 输入像素格式：RGB24
                '-s', f'{width}x{height}',   # 视频尺寸
                '-r', str(fps),             # 帧率
                '-i', 'pipe:0',             # 输入：管道
                '-c:v', 'libx264',          # 视频编码器：H.264
                '-crf', str(crf),           # 视频质量
                '-preset', 'ultrafast',     # 编码预设：最快
                '-pix_fmt', 'yuv420p',      # 输出像素格式：YUV420P（MP4兼容）
                '-profile:v', 'baseline',   # H.264配置文件：baseline（最大兼容性）
                '-level', '3.0',            # H.264级别：3.0
                '-movflags', '+faststart',  # MP4优化：快速启动
                '-g', '10',                 # 关键帧间隔：10帧
                '-vf', 'scale=out_color_matrix=bt709',  # 视频滤镜：颜色矩阵转换
                '-color_range', 'tv',       # 颜色范围
                '-colorspace', 'bt709',     # 颜色空间
                '-color_primaries', 'bt709', # 颜色原色
                '-color_trc', 'bt709',      # 颜色传输特性
                output_path
            ]
            
            print(f"[PIPE] FFmpeg命令: {' '.join(cmd)}")
            
            # 通过管道传递数据
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=frame_data)
            
            if process.returncode == 0:
                print("[PIPE] 成功使用FFmpeg管道创建视频")
                return output_path
            else:
                print(f"[PIPE] FFmpeg管道创建视频失败: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg错误: {stderr.decode()}")
                
        except FileNotFoundError:
            print("[PIPE] FFmpeg不可用")
            raise RuntimeError("无法创建视频：FFmpeg不可用")
        except Exception as e:
            print(f"[PIPE] FFmpeg管道创建视频异常: {e}")
            raise RuntimeError(f"视频创建失败: {e}")

    def _create_video_with_audio_pipe(self, images: List[np.ndarray], output_path: str, fps: float, width: int, height: int, crf: int = 19, audio=None) -> str:
        """使用FFmpeg管道一步完成音视频合成（类似ComfyUI-VideoHelperSuite）"""
        try:
            print(f"[AUDIO_PIPE] 开始一步完成音视频合成，帧数: {len(images)}, 尺寸: {width}x{height}, FPS: {fps}, CRF: {crf}")
            
            # 处理音频输入
            audio_path = self._process_audio_input(audio)
            print(f"[AUDIO_PIPE] 音频文件准备完成: {audio_path}")
            
            # 准备图像数据
            frame_data = b''
            for i, img in enumerate(images):
                # 确保图像是RGB格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = img
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    # RGBA转RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:
                    # 灰度图转RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # 确保尺寸正确
                if img_rgb.shape[1] != width or img_rgb.shape[0] != height:
                    img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # 确保数据类型正确
                if img_rgb.dtype != np.uint8:
                    img_rgb = img_rgb.astype(np.uint8)
                
                # 转换为字节数据
                frame_bytes = img_rgb.tobytes()
                frame_data += frame_bytes
                
                if i % 10 == 0:  # 每10帧打印一次进度
                    print(f"[AUDIO_PIPE] 已处理 {i+1}/{len(images)} 帧")
            
            print(f"[AUDIO_PIPE] 图像数据准备完成，总大小: {len(frame_data)} 字节")
            
            # 使用FFmpeg一步完成音视频合成（修复MP4兼容性问题）
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',           # 输入格式：原始视频
                '-pix_fmt', 'rgb24',        # 输入像素格式：RGB24
                '-s', f'{width}x{height}',   # 视频尺寸
                '-r', str(fps),             # 帧率
                '-i', 'pipe:0',             # 视频输入：管道
                '-i', audio_path,           # 音频输入：文件
                '-c:v', 'libx264',          # 视频编码器：H.264
                '-crf', str(crf),           # 视频质量
                '-preset', 'ultrafast',     # 编码预设：最快
                '-pix_fmt', 'yuv420p',      # 输出像素格式：YUV420P（MP4兼容）
                '-profile:v', 'baseline',   # H.264配置文件：baseline（最大兼容性）
                '-level', '3.0',            # H.264级别：3.0
                '-movflags', '+faststart',  # MP4优化：快速启动
                '-g', '10',                 # 关键帧间隔：10帧
                '-vf', 'scale=out_color_matrix=bt709',  # 视频滤镜：颜色矩阵转换
                '-color_range', 'tv',       # 颜色范围
                '-colorspace', 'bt709',     # 颜色空间
                '-color_primaries', 'bt709', # 颜色原色
                '-color_trc', 'bt709',      # 颜色传输特性
                '-c:a', 'aac',              # 音频编码器：AAC
                '-map', '0:v:0',            # 映射视频流
                '-map', '1:a:0',            # 映射音频流
                '-shortest',                # 以最短流为准
                output_path
            ]
            
            print(f"[AUDIO_PIPE] FFmpeg命令: {' '.join(cmd)}")
            
            # 通过管道传递数据
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=frame_data)
            
            if process.returncode == 0:
                print("[AUDIO_PIPE] 成功一步完成音视频合成")
                return output_path
            else:
                print(f"[AUDIO_PIPE] FFmpeg音视频合成失败: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg错误: {stderr.decode()}")
                
        except FileNotFoundError:
            print("[AUDIO_PIPE] FFmpeg不可用")
            raise RuntimeError("无法创建音视频：FFmpeg不可用")
        except Exception as e:
            print(f"[AUDIO_PIPE] FFmpeg音视频合成异常: {e}")
            raise RuntimeError(f"音视频合成失败: {e}")
        finally:
            # 清理音频文件
            try:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"[AUDIO_PIPE] 已清理音频文件: {audio_path}")
            except Exception as e:
                print(f"[AUDIO_PIPE] 清理音频文件失败: {e}")

    def _create_video_with_ffmpeg(self, images: List[np.ndarray], output_path: str, fps: float, width: int, height: int, crf: int = 19) -> str:
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
                '-crf', str(crf),  # 可配置压缩率
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
    
    def images_to_mp4_with_audio(self, images: List[np.ndarray], fps: float, audio, crf: int = 19) -> str:
        """将多张图片转换为带音频的MP4视频（现在调用统一方法）"""
        print("[COMPAT] 调用兼容性方法，将转发到统一方法")
        return self.images_to_mp4(images, fps, crf, audio)
    
    def _process_audio_input(self, audio) -> str:
        """简化的音频输入处理（类似ComfyUI-VideoHelperSuite）"""
        random_suffix = str(uuid.uuid4())[:8]
        audio_path = os.path.join(self.temp_dir, f"temp_audio_{random_suffix}.wav")
        
        try:
            # 1. 文件路径（最常见）
            if isinstance(audio, str) and os.path.exists(audio):
                import shutil
                shutil.copy2(audio, audio_path)
                print(f"[AUDIO] 复制音频文件: {audio} -> {audio_path}")
                return audio_path
            
            # 2. ComfyUI标准格式（最常见）
            elif isinstance(audio, dict) and 'samples' in audio:
                audio_data = audio['samples']
                sample_rate = audio.get('sample_rate', 44100)
                
                if hasattr(audio_data, 'cpu'):
                    audio_data = audio_data.cpu().numpy()
                
                import soundfile as sf
                sf.write(audio_path, audio_data, sample_rate)
                print(f"[AUDIO] 保存ComfyUI音频数据: {len(audio_data)} 样本, {sample_rate}Hz")
                return audio_path
            
            # 3. LazyAudioMap（ComfyUI-VideoHelperSuite格式）
            elif hasattr(audio, 'file') and audio.file:
                if os.path.exists(audio.file):
                    import shutil
                    shutil.copy2(audio.file, audio_path)
                    print(f"[AUDIO] 复制LazyAudioMap文件: {audio.file} -> {audio_path}")
                    return audio_path
                else:
                    print(f"[AUDIO] LazyAudioMap文件不存在: {audio.file}")
            
            # 4. 直接音频数据
            elif isinstance(audio, np.ndarray):
                import soundfile as sf
                sf.write(audio_path, audio, 44100)
                print(f"[AUDIO] 保存numpy音频数据: {len(audio)} 样本")
                return audio_path
            
            # 5. torch张量
            elif hasattr(audio, 'cpu'):
                audio_np = audio.cpu().numpy()
                import soundfile as sf
                sf.write(audio_path, audio_np, 44100)
                print(f"[AUDIO] 保存torch音频数据: {len(audio_np)} 样本")
                return audio_path
            
            else:
                raise ValueError(f"不支持的音频格式: {type(audio)}，支持的格式：str, dict, np.ndarray, torch.Tensor")
                
        except Exception as e:
            raise RuntimeError(f"音频处理失败: {e}")
    
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
