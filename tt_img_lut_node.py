import numpy as np
import torch
import os
from typing import List
import cv2

class TTImgLUTNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lut_file": ("STRING", {"default": "", "multiline": False}),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "LUT色彩查找表节点\n支持.cube格式的LUT文件\nLUT强度：0.0到1.0（1.0为完全应用）\n适用于电影级调色和专业色彩校正\n请确保LUT文件路径正确", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"
    CATEGORY = "TT Tools"
    
    def apply_lut(self, images, lut_file="", lut_strength=1.0, usage_notes=None):
        """
        对输入图像应用LUT色彩查找表
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            lut_file: LUT文件路径
            lut_strength: LUT应用强度 (0.0 到 1.0，1.0为完全应用)
            usage_notes: 使用说明
        
        Returns:
            tuple: 处理后的图像张量
        """
        try:
            # 检查LUT文件是否存在
            if not lut_file or not os.path.exists(lut_file):
                print(f"Warning: LUT file not found: {lut_file}")
                return (images,)
            
            # 加载LUT
            lut_table = self._load_lut_file(lut_file)
            if lut_table is None:
                print(f"Error: Failed to load LUT file: {lut_file}")
                return (images,)
            
            # 将ComfyUI的torch张量转换为numpy数组进行处理
            numpy_images = []
            
            for img in images:
                if hasattr(img, 'cpu'):
                    # 如果是torch张量，转换为numpy
                    img_np = img.cpu().numpy()
                    # 确保值范围在0-1之间（ComfyUI标准格式）
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                else:
                    # 如果已经是numpy数组
                    img_np = np.array(img)
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                
                # 应用LUT
                lut_applied_img = self._apply_lut_to_image(img_np, lut_table, lut_strength)
                numpy_images.append(lut_applied_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image LUT 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"LUT文件: {os.path.basename(lut_file)}")
                print(f"LUT强度: {lut_strength}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image LUT node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _load_lut_file(self, lut_file_path: str) -> np.ndarray:
        """
        加载LUT文件
        
        Args:
            lut_file_path: LUT文件路径
        
        Returns:
            np.ndarray: LUT查找表，形状为(64, 64, 64, 3)或None
        """
        try:
            file_ext = os.path.splitext(lut_file_path)[1].lower()
            
            if file_ext == '.cube':
                return self._load_cube_lut(lut_file_path)
            else:
                print(f"Unsupported LUT file format: {file_ext}")
                return None
                
        except Exception as e:
            print(f"Error loading LUT file: {str(e)}")
            return None
    
    def _load_cube_lut(self, cube_file_path: str) -> np.ndarray:
        """
        加载.cube格式的LUT文件
        
        Args:
            cube_file_path: .cube文件路径
        
        Returns:
            np.ndarray: LUT查找表，形状为(64, 64, 64, 3)
        """
        try:
            with open(cube_file_path, 'r') as f:
                lines = f.readlines()
            
            # 解析LUT_SIZE
            lut_size = 64  # 默认大小
            for line in lines:
                if line.strip().startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[-1])
                    break
            
            # 查找数据开始位置
            data_start = -1
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    try:
                        # 尝试解析第一行数据
                        values = line.strip().split()
                        if len(values) == 3:
                            float(values[0])
                            float(values[1])
                            float(values[2])
                            data_start = i
                            break
                    except:
                        continue
            
            if data_start == -1:
                print("Error: Could not find LUT data in cube file")
                return None
            
            # 读取LUT数据
            lut_data = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('#'):
                    values = line.split()
                    if len(values) == 3:
                        try:
                            r, g, b = float(values[0]), float(values[1]), float(values[2])
                            lut_data.append([r, g, b])
                        except:
                            continue
            
            if len(lut_data) != lut_size ** 3:
                print(f"Error: Expected {lut_size**3} entries, got {len(lut_data)}")
                return None
            
            # 重塑为3D LUT
            lut_table = np.array(lut_data).reshape(lut_size, lut_size, lut_size, 3)
            
            return lut_table
            
        except Exception as e:
            print(f"Error loading cube LUT: {str(e)}")
            return None
    
    def _apply_lut_to_image(self, image: np.ndarray, lut_table: np.ndarray, strength: float) -> np.ndarray:
        """
        对图像应用LUT
        
        Args:
            image: 输入图像数组 (height, width, channels)，值范围0-1
            lut_table: LUT查找表，形状为(size, size, size, 3)
            strength: LUT应用强度 (0.0 到 1.0)
        
        Returns:
            np.ndarray: 处理后的图像数组，值范围0-1
        """
        try:
            # 确保图像是3通道RGB格式
            if len(image.shape) == 2:
                # 灰度图转RGB
                image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) / 255.0
            elif image.shape[2] == 4:
                # RGBA转RGB
                image = image[:, :, :3]
            elif image.shape[2] == 1:
                # 单通道转RGB
                image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) / 255.0
            
            # 获取LUT尺寸
            lut_size = lut_table.shape[0]
            
            # 将图像值映射到LUT索引
            # 图像值范围0-1，映射到0-(lut_size-1)
            image_indices = (image * (lut_size - 1)).astype(np.float32)
            
            # 使用三线性插值应用LUT
            result = self._trilinear_interpolation(image_indices, lut_table)
            
            # 应用强度混合
            if strength < 1.0:
                result = image * (1.0 - strength) + result * strength
            
            # 确保值在有效范围内
            result = np.clip(result, 0.0, 1.0)
            
            return result
                
        except Exception as e:
            print(f"Error applying LUT: {str(e)}")
            return image
    
    def _trilinear_interpolation(self, indices: np.ndarray, lut_table: np.ndarray) -> np.ndarray:
        """
        三线性插值应用LUT
        
        Args:
            indices: 图像索引数组，形状为(height, width, 3)
            lut_table: LUT查找表，形状为(size, size, size, 3)
        
        Returns:
            np.ndarray: 插值后的图像数组
        """
        try:
            height, width, _ = indices.shape
            lut_size = lut_table.shape[0]
            
            # 获取整数部分和小数部分
            x0 = np.floor(indices[:, :, 0]).astype(np.int32)
            y0 = np.floor(indices[:, :, 1]).astype(np.int32)
            z0 = np.floor(indices[:, :, 2]).astype(np.int32)
            
            xd = indices[:, :, 0] - x0
            yd = indices[:, :, 1] - y0
            zd = indices[:, :, 2] - z0
            
            # 确保索引在有效范围内
            x0 = np.clip(x0, 0, lut_size - 2)
            y0 = np.clip(y0, 0, lut_size - 2)
            z0 = np.clip(z0, 0, lut_size - 2)
            
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1
            
            # 获取8个顶点的值
            c000 = lut_table[x0, y0, z0]
            c001 = lut_table[x0, y0, z1]
            c010 = lut_table[x0, y1, z0]
            c011 = lut_table[x0, y1, z1]
            c100 = lut_table[x1, y0, z0]
            c101 = lut_table[x1, y0, z1]
            c110 = lut_table[x1, y1, z0]
            c111 = lut_table[x1, y1, z1]
            
            # 三线性插值
            c00 = c000 * (1 - xd[..., np.newaxis]) + c100 * xd[..., np.newaxis]
            c01 = c001 * (1 - xd[..., np.newaxis]) + c101 * xd[..., np.newaxis]
            c10 = c010 * (1 - xd[..., np.newaxis]) + c110 * xd[..., np.newaxis]
            c11 = c011 * (1 - xd[..., np.newaxis]) + c111 * xd[..., np.newaxis]
            
            c0 = c00 * (1 - yd[..., np.newaxis]) + c10 * yd[..., np.newaxis]
            c1 = c01 * (1 - yd[..., np.newaxis]) + c11 * yd[..., np.newaxis]
            
            result = c0 * (1 - zd[..., np.newaxis]) + c1 * zd[..., np.newaxis]
            
            return result
            
        except Exception as e:
            print(f"Error in trilinear interpolation: {str(e)}")
            return indices

# 节点类定义完成
