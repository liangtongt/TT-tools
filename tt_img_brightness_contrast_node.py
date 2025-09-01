import numpy as np
import torch
from typing import List

class TTImgBrightnessContrastNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "图像亮度和对比度调节节点\n亮度调节范围：-1.0 到 1.0（0.0为原始亮度）\n对比度调节范围：0.0 到 3.0（1.0为原始对比度）\n适用于图像增强和视觉效果调整", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_brightness_contrast"
    CATEGORY = "TT Tools"
    
    def adjust_brightness_contrast(self, images, brightness=0.0, contrast=1.0, usage_notes=None):
        """
        调节输入图像的亮度和对比度
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            brightness: 亮度调节值 (-1.0 到 1.0，0.0为原始亮度)
            contrast: 对比度调节值 (0.0 到 3.0，1.0为原始对比度)
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
                
                # 应用亮度和对比度调节操作
                adjusted_img = self._apply_brightness_contrast(img_np, brightness, contrast)
                numpy_images.append(adjusted_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image Brightness Contrast 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"亮度调节: {brightness}")
                print(f"对比度调节: {contrast}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image Brightness Contrast node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_brightness_contrast(self, image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """
        对单张图像应用亮度和对比度调节操作
        
        Args:
            image: 输入图像数组 (height, width, channels)
            brightness: 亮度调节值 (-1.0 到 1.0)
            contrast: 对比度调节值 (0.0 到 3.0)
        
        Returns:
            np.ndarray: 处理后的图像数组
        """
        try:
            # 复制图像以避免修改原始数据
            result = image.copy()
            
            # 应用对比度调节
            # 对比度公式：new_value = (old_value - 0.5) * contrast + 0.5
            # 这样可以保持中值(0.5)不变，向两端拉伸或压缩
            result = (result - 0.5) * contrast + 0.5
            
            # 应用亮度调节
            # 亮度公式：new_value = old_value + brightness
            result = result + brightness
            
            # 确保值在有效范围内 [0, 1]
            result = np.clip(result, 0.0, 1.0)
            
            return result
                
        except Exception as e:
            print(f"Error in _apply_brightness_contrast: {str(e)}")
            return image

# 节点类定义完成
