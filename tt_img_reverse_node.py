import numpy as np
import torch
from typing import List

class TTImgReverseNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reverse_type": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "图像反向节点\n支持水平翻转、垂直翻转和同时翻转\n适用于图像数据增强和特殊效果处理", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverse_images"
    CATEGORY = "TT Tools"
    
    def reverse_images(self, images, reverse_type="horizontal", usage_notes=None):
        """
        对输入的图像进行反向处理
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            reverse_type: 翻转类型 ("horizontal", "vertical", "both")
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
                
                # 应用反向操作
                reversed_img = self._apply_reverse(img_np, reverse_type)
                numpy_images.append(reversed_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image Reverse 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"翻转类型: {reverse_type}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image Reverse node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_reverse(self, image: np.ndarray, reverse_type: str) -> np.ndarray:
        """
        对单张图像应用反向操作
        
        Args:
            image: 输入图像数组 (height, width, channels)
            reverse_type: 翻转类型
        
        Returns:
            np.ndarray: 处理后的图像数组
        """
        try:
            if reverse_type == "horizontal":
                # 水平翻转（左右翻转）
                return np.fliplr(image)
            elif reverse_type == "vertical":
                # 垂直翻转（上下翻转）
                return np.flipud(image)
            elif reverse_type == "both":
                # 同时进行水平和垂直翻转
                return np.flipud(np.fliplr(image))
            else:
                # 默认返回原图
                return image
                
        except Exception as e:
            print(f"Error in _apply_reverse: {str(e)}")
            return image

# 节点类定义完成
