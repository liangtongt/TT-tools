import numpy as np
import torch
import os
from typing import List
import cv2

class TTImgLUTNode:
    def __init__(self):
        pass
    
    @classmethod
    def _get_default_lut(cls):
        """
        生成默认的复古暖色调LUT内容
        
        Returns:
            str: .cube格式的LUT内容
        """
        # 创建一个32x32x32的复古暖色调LUT
        lut_size = 32
        lut_data = []
        
        # 生成复古暖色调LUT数据
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    # 将索引转换为0-1范围
                    r_val = r / (lut_size - 1)
                    g_val = g / (lut_size - 1)
                    b_val = b / (lut_size - 1)
                    
                    # 应用复古暖色调效果
                    # 增强红色和黄色，降低蓝色
                    new_r = min(1.0, r_val * 1.2 + 0.1)  # 增强红色
                    new_g = min(1.0, g_val * 1.1 + 0.05)  # 轻微增强绿色
                    new_b = max(0.0, b_val * 0.8 - 0.1)   # 降低蓝色
                    
                    # 添加复古色调（偏黄）
                    new_r = min(1.0, new_r + 0.05)
                    new_g = min(1.0, new_g + 0.03)
                    
                    # 添加轻微的对比度增强
                    new_r = (new_r - 0.5) * 1.1 + 0.5
                    new_g = (new_g - 0.5) * 1.1 + 0.5
                    new_b = (new_b - 0.5) * 1.1 + 0.5
                    
                    # 确保值在0-1范围内
                    new_r = max(0.0, min(1.0, new_r))
                    new_g = max(0.0, min(1.0, new_g))
                    new_b = max(0.0, min(1.0, new_b))
                    
                    lut_data.append(f"{new_r:.6f} {new_g:.6f} {new_b:.6f}")
        
        # 构建.cube格式内容
        cube_content = f"""TITLE "TT Retro Warm LUT"
LUT_3D_SIZE {lut_size}
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

"""
        cube_content += "\n".join(lut_data)
        
        return cube_content
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lut_content": ("STRING", {"default": cls._get_default_lut(), "multiline": True}),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "LUT色彩查找表节点\n支持.cube格式的LUT文件内容\nLUT强度：0.0到1.0（1.0为完全应用）\n适用于电影级调色和专业色彩校正\n默认提供复古暖色调LUT效果", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"
    CATEGORY = "TT Tools"
    
    def apply_lut(self, images, lut_content="", lut_strength=1.0, usage_notes=None):
        """
        对输入图像应用LUT色彩查找表
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            lut_content: LUT文件内容（.cube格式）
            lut_strength: LUT应用强度 (0.0 到 1.0，1.0为完全应用)
            usage_notes: 使用说明
        
        Returns:
            tuple: 处理后的图像张量
        """
        try:
            # 检查LUT内容是否为空
            if not lut_content or not lut_content.strip():
                print("Warning: LUT content is empty")
                return (images,)
            
            # 加载LUT
            lut_table = self._load_lut_content(lut_content)
            if lut_table is None:
                print("Error: Failed to load LUT content")
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
                print(f"LUT内容长度: {len(lut_content)} 字符")
                print(f"LUT强度: {lut_strength}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image LUT node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _load_lut_content(self, lut_content: str) -> np.ndarray:
        """
        加载LUT内容
        
        Args:
            lut_content: LUT文件内容字符串
        
        Returns:
            np.ndarray: LUT查找表，形状为(64, 64, 64, 3)或None
        """
        try:
            # 检查内容是否包含.cube格式的特征
            if 'LUT_3D_SIZE' in lut_content or lut_content.strip().startswith('TITLE'):
                return self._load_cube_lut_content(lut_content)
            else:
                print("Unsupported LUT content format. Please provide .cube format content.")
                return None
                
        except Exception as e:
            print(f"Error loading LUT content: {str(e)}")
            return None
    
    def _load_cube_lut_content(self, cube_content: str) -> np.ndarray:
        """
        加载.cube格式的LUT内容
        
        Args:
            cube_content: .cube文件内容字符串
        
        Returns:
            np.ndarray: LUT查找表，形状为(64, 64, 64, 3)
        """
        try:
            lines = cube_content.strip().split('\n')
            
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
                print("Error: Could not find LUT data in cube content")
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
            print(f"Error loading cube LUT content: {str(e)}")
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
