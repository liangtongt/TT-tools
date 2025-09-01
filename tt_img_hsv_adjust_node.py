import numpy as np
import torch
from typing import List
import cv2

class TTImgHSVAdjustNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "HSV色彩调节节点\n色相调节：-180°到180°（0°为原始色相）\n饱和度调节：0.0到3.0（1.0为原始饱和度）\n明度调节：0.0到3.0（1.0为原始明度）\n适用于色彩风格转换和艺术效果制作", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_hsv"
    CATEGORY = "TT Tools"
    
    def adjust_hsv(self, images, hue_shift=0.0, saturation=1.0, value=1.0, usage_notes=None):
        """
        调节输入图像的色相、饱和度、明度
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            hue_shift: 色相偏移值 (-180° 到 180°，0°为原始色相)
            saturation: 饱和度调节值 (0.0 到 3.0，1.0为原始饱和度)
            value: 明度调节值 (0.0 到 3.0，1.0为原始明度)
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
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = (img_np * 255).astype(np.uint8)
                else:
                    # 如果已经是numpy数组
                    img_np = np.array(img)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                
                # 应用HSV调节操作
                adjusted_img = self._apply_hsv_adjustment(img_np, hue_shift, saturation, value)
                numpy_images.append(adjusted_img)
            
            # 转换回torch张量，确保值范围在0-1之间
            output_tensor = torch.from_numpy(np.array(numpy_images)).float() / 255.0
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image HSV Adjust 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"色相偏移: {hue_shift}°")
                print(f"饱和度调节: {saturation}")
                print(f"明度调节: {value}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image HSV Adjust node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_hsv_adjustment(self, image: np.ndarray, hue_shift: float, saturation: float, value: float) -> np.ndarray:
        """
        对单张图像应用HSV调节操作
        
        Args:
            image: 输入图像数组 (height, width, channels)，值范围0-255
            hue_shift: 色相偏移值 (-180° 到 180°)
            saturation: 饱和度调节值 (0.0 到 3.0)
            value: 明度调节值 (0.0 到 3.0)
        
        Returns:
            np.ndarray: 处理后的图像数组，值范围0-255
        """
        try:
            # 确保图像是3通道RGB格式
            if len(image.shape) == 2:
                # 灰度图转RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA转RGB
                image = image[:, :, :3]
            elif image.shape[2] == 1:
                # 单通道转RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 转换RGB到HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 应用色相偏移
            if hue_shift != 0.0:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
            
            # 应用饱和度调节
            if saturation != 1.0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            
            # 应用明度调节
            if value != 1.0:
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
            
            # 转换HSV回RGB
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return result
                
        except Exception as e:
            print(f"Error in _apply_hsv_adjustment: {str(e)}")
            return image

# 节点类定义完成
