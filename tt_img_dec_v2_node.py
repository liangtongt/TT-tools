import os
from PIL import Image
import numpy as np
import torch
from typing import Tuple, Optional
import cv2


class TTImgDecV2Node:
    def __init__(self):
        import folder_paths
        try:
            if hasattr(folder_paths, 'get_output_directory'):
                self.output_dir = folder_paths.get_output_directory()
            elif hasattr(folder_paths, 'output_directory'):
                self.output_dir = folder_paths.output_directory
            else:
                self.output_dir = "output"
        except Exception:
            self.output_dir = "output"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            self.output_dir = "output"
            os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_filename": ("STRING", {"default": "tt_img_dec_file", "multiline": False}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "用于解码 TT img enc V2 图片（自适应跳过比例，扫描MAGIC定位）。输出与V1一致。", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "audio", "file_path", "fps")
    FUNCTION = "extract_file_from_image"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True

    def extract_file_from_image(self, image, output_filename="tt_img_dec_file", usage_notes=None):
        try:
            if hasattr(image, 'cpu'):
                img_np = image.cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                img_np = np.array(image).astype(np.uint8)
            if len(img_np.shape) == 4:
                img_np = img_np[0]

            file_data, file_extension = self._extract_v2_file_data(img_np)
            if file_data is None:
                return (None, None, "", 0.0)

            if not output_filename:
                output_filename = "tt_img_dec_file"
            if not output_filename.endswith(f".{file_extension}"):
                output_filename = f"{output_filename}.{file_extension}"

            base_name = os.path.splitext(output_filename)[0]
            extension = os.path.splitext(output_filename)[1]
            counter = 1
            final_filename = output_filename
            while os.path.exists(os.path.join(self.output_dir, final_filename)):
                final_filename = f"{base_name}_{counter}{extension}"
                counter += 1
            output_path = os.path.join(self.output_dir, final_filename)

            with open(output_path, 'wb') as f:
                f.write(file_data)

            output_image, output_audio, fps = self._process_decoded_file(output_path, file_extension)
            return (output_image, output_audio, output_path, fps)

        except Exception as e:
            print(f"解码失败(V2): {e}")
            return (None, None, "", 0.0)

    # ===== V2 提取核心 =====

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

    def _build_flat_rgb(self, arr: np.ndarray) -> np.ndarray:
        h, w, ch = arr.shape
        if ch >= 3:
            rgb = arr[:, :, :3]
            return rgb.transpose(0, 1, 2).reshape(-1).astype(np.uint8)
        return arr.transpose(0, 1, 2).reshape(-1).astype(np.uint8)

    def _extract_v2_file_data(self, image_array: np.ndarray) -> tuple:
        if len(image_array.shape) != 3 or image_array.shape[2] not in [3, 4]:
            return None, None
        height, width, channels = image_array.shape
        scan_rows = max(1, int(height * 0.15))
        flat = self._build_flat_rgb(image_array[:scan_rows])

        magic = b'TTv2'
        m0, m1, m2, m3 = magic[0], magic[1], magic[2], magic[3]
        L = len(flat)
        i = 0
        while i + 4 <= L:
            if flat[i] == m0 and flat[i+1] == m1 and flat[i+2] == m2 and flat[i+3] == m3:
                if i + 10 > L:
                    break
                b4 = int(flat[i+4]); b5 = int(flat[i+5]); b6 = int(flat[i+6]); b7 = int(flat[i+7])
                hdr_len = ((b4 << 24) | (b5 << 16) | (b6 << 8) | b7)
                c8 = int(flat[i+8]); c9 = int(flat[i+9])
                crc = ((c8 << 8) | c9)
                total_needed = 10 + hdr_len

                # 从整幅拼接以确保足够数据
                full = self._build_flat_rgb(image_array)
                if hdr_len < 0 or i + total_needed > len(full):
                    i += 1
                    continue
                packet = bytes(full[i:i+total_needed])

                inner = packet[10:]
                if self._crc16_ccitt(inner) != crc:
                    i += 1
                    continue

                if len(inner) < 1 + 1 + 1 + 4:
                    i += 1
                    continue
                version = inner[0]
                # flags = inner[1]  # 如需可输出
                ext_len = inner[2]
                if len(inner) < 3 + ext_len + 4:
                    i += 1
                    continue
                ext = inner[3:3+ext_len].decode('utf-8', errors='ignore')
                data_len = int.from_bytes(inner[3+ext_len:3+ext_len+4], 'big')
                data = inner[3+ext_len+4:]
                if version != 2 or data_len < 0 or len(data) < data_len:
                    i += 1
                    continue
                return bytes(data[:data_len]), ext
            i += 1
        return None, None

    # ===== 后处理与辅助 =====

    def _process_decoded_file(self, file_path: str, file_extension: str) -> Tuple[Optional[torch.Tensor], Optional[dict], float]:
        try:
            if file_extension.lower() in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']:
                image_result = self._process_image_file(file_path)
                return image_result, None, 0.0
            elif file_extension.lower() in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                return self._process_video_file(file_path)
            elif file_extension.lower() in ['wav', 'mp3', 'aac', 'flac', 'ogg', 'm4a']:
                audio_result = self._process_audio_file(file_path)
                return None, audio_result, 0.0
            else:
                print(f"不支持的文件类型: {file_extension}")
                return None, None, 0.0
        except Exception as e:
            print(f"处理解码文件失败: {e}")
            return None, None, 0.0

    def _process_image_file(self, file_path: str) -> Optional[torch.Tensor]:
        try:
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"处理图片文件失败: {e}")
            return None

    def _process_video_file(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[dict], float]:
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {file_path}")
                return None, None, 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
            if not frames:
                print("无法读取视频帧")
                return None, None, 0.0
            frames_array = np.array(frames).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(frames_array)
            audio_dict = self._extract_audio_from_video(file_path)
            return img_tensor, audio_dict, fps
        except Exception as e:
            print(f"处理视频文件失败: {e}")
            return None, None, 0.0

    def _process_audio_file(self, file_path: str) -> Optional[dict]:
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(file_path)
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            if len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.transpose(0, 1).unsqueeze(0)
            elif len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            return {
                'waveform': audio_tensor,
                'sample_rate': sample_rate
            }
        except Exception as e:
            print(f"处理音频文件失败: {e}")
            return None

    def _extract_audio_from_video(self, video_path: str) -> Optional[dict]:
        try:
            import subprocess
            import tempfile
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2', '-y', temp_audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg提取音频失败: {result.stderr}")
                return None
            audio_dict = self._process_audio_file(temp_audio_path)
            os.unlink(temp_audio_path)
            return audio_dict
        except Exception as e:
            print(f"从视频提取音频失败: {e}")
            return None


