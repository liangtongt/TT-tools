import numpy as np
import torch
from typing import List

class TTImgGrayscaleNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grayscale_method": (["luminance", "average", "red", "green", "blue", "max", "min"], {"default": "luminance"}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "图像转灰度节点\n支持多种灰度转换方法：\n- luminance: 亮度加权法（推荐）\n- average: 平均值法\n- red/green/blue: 单通道法\n- max/min: 最大值/最小值法", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "TT Tools"
    
    def convert_to_grayscale(self, images, grayscale_method="luminance", usage_notes=None):
        """
        将输入的图像转换为灰度图像
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            grayscale_method: 灰度转换方法
            usage_notes: 使用说明
        
        Returns:
            tuple: 处理后的灰度图像张量
        """
        try:
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
                
                # 应用灰度转换操作
                grayscale_img = self._apply_grayscale_conversion(img_np, grayscale_method)
                numpy_images.append(grayscale_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image Grayscale 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"转换方法: {grayscale_method}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image Grayscale node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_grayscale_conversion(self, image: np.ndarray, method: str) -> np.ndarray:
        """
        对单张图像应用灰度转换操作
        
        Args:
            image: 输入图像数组 (height, width, channels)
            method: 灰度转换方法
        
        Returns:
            np.ndarray: 处理后的灰度图像数组 (height, width, 1)
        """
        try:
            # 确保图像是3维的
            if len(image.shape) == 2:
                # 如果是2维灰度图，直接返回
                return np.expand_dims(image, axis=2)
            
            # 获取图像尺寸
            height, width, channels = image.shape
            
            # 如果是单通道图像，直接返回
            if channels == 1:
                return image
            
            # 提取RGB通道（忽略alpha通道）
            if channels >= 3:
                r = image[:, :, 0]
                g = image[:, :, 1]
                b = image[:, :, 2]
            else:
                # 如果通道数不足3，使用第一个通道
                r = g = b = image[:, :, 0]
            
            # 根据方法进行灰度转换
            if method == "luminance":
                # 亮度加权法（ITU-R BT.709标准）
                # 人眼对绿色最敏感，红色次之，蓝色最不敏感
                grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b
                
            elif method == "average":
                # 平均值法
                grayscale = (r + g + b) / 3.0
                
            elif method == "red":
                # 红色通道法
                grayscale = r
                
            elif method == "green":
                # 绿色通道法
                grayscale = g
                
            elif method == "blue":
                # 蓝色通道法
                grayscale = b
                
            elif method == "max":
                # 最大值法
                grayscale = np.maximum(np.maximum(r, g), b)
                
            elif method == "min":
                # 最小值法
                grayscale = np.minimum(np.minimum(r, g), b)
                
            else:
                # 默认使用亮度加权法
                grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b
            
            # 确保值在有效范围内
            grayscale = np.clip(grayscale, 0.0, 1.0)
            
            # 转换为3通道格式以保持与ComfyUI的兼容性
            # 将灰度值复制到所有3个通道
            grayscale_3ch = np.stack([grayscale, grayscale, grayscale], axis=2)
            
            return grayscale_3ch
                
        except Exception as e:
            print(f"Error in _apply_grayscale_conversion: {str(e)}")
            return image

# 节点类定义完成
