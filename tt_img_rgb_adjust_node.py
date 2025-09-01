import numpy as np
import torch
from typing import List

class TTImgRGBAdjustNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "red_adjust": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "green_adjust": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "blue_adjust": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "RGB三色相调节节点\n支持独立调节红、绿、蓝三个通道\n调节范围：-1.0 到 1.0（0.0为原始值）\n正值增强该颜色，负值减弱该颜色\n适用于色彩校正和艺术效果制作", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_rgb_channels"
    CATEGORY = "TT Tools"
    
    def adjust_rgb_channels(self, images, red_adjust=0.0, green_adjust=0.0, blue_adjust=0.0, usage_notes=None):
        """
        调节输入图像的RGB三色相
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            red_adjust: 红色通道调节值 (-1.0 到 1.0，0.0为原始值)
            green_adjust: 绿色通道调节值 (-1.0 到 1.0，0.0为原始值)
            blue_adjust: 蓝色通道调节值 (-1.0 到 1.0，0.0为原始值)
            usage_notes: 使用说明
        
        Returns:
            tuple: 处理后的图像张量
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
                
                # 应用RGB通道调节操作
                adjusted_img = self._apply_rgb_adjustment(img_np, red_adjust, green_adjust, blue_adjust)
                numpy_images.append(adjusted_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image RGB Adjust 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"红色通道调节: {red_adjust}")
                print(f"绿色通道调节: {green_adjust}")
                print(f"蓝色通道调节: {blue_adjust}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image RGB Adjust node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_rgb_adjustment(self, image: np.ndarray, red_adjust: float, green_adjust: float, blue_adjust: float) -> np.ndarray:
        """
        对单张图像应用RGB通道调节操作
        
        Args:
            image: 输入图像数组 (height, width, channels)
            red_adjust: 红色通道调节值 (-1.0 到 1.0)
            green_adjust: 绿色通道调节值 (-1.0 到 1.0)
            blue_adjust: 蓝色通道调节值 (-1.0 到 1.0)
        
        Returns:
            np.ndarray: 处理后的图像数组
        """
        try:
            # 复制图像以避免修改原始数据
            result = image.copy()
            
            # 获取图像尺寸
            height, width, channels = image.shape
            
            # 确保至少有3个通道（RGB）
            if channels >= 3:
                # 应用红色通道调节
                result[:, :, 0] = result[:, :, 0] + red_adjust
                
                # 应用绿色通道调节
                result[:, :, 1] = result[:, :, 1] + green_adjust
                
                # 应用蓝色通道调节
                result[:, :, 2] = result[:, :, 2] + blue_adjust
                
                # 如果有alpha通道，保持不变
                if channels == 4:
                    result[:, :, 3] = image[:, :, 3]
            else:
                # 如果通道数不足3，对所有通道应用相同的调节
                # 使用平均值作为调节值
                avg_adjust = (red_adjust + green_adjust + blue_adjust) / 3.0
                result = result + avg_adjust
            
            # 确保值在有效范围内 [0, 1]
            result = np.clip(result, 0.0, 1.0)
            
            return result
                
        except Exception as e:
            print(f"Error in _apply_rgb_adjustment: {str(e)}")
            return image

# 节点类定义完成
