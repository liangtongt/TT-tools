import numpy as np
import torch
from typing import List

class TTImgColorReverseNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reverse_mode": (["full", "rgb_only", "preserve_alpha"], {"default": "full"}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "图像颜色反向节点\n支持完全反转、仅RGB反转、保留透明度反转\n适用于创建负片效果和特殊视觉效果", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverse_colors"
    CATEGORY = "TT Tools"
    
    def reverse_colors(self, images, reverse_mode="full", usage_notes=None):
        """
        对输入的图像进行颜色反向处理
        
        Args:
            images: 输入的图像张量 (batch_size, height, width, channels)
            reverse_mode: 反转模式 ("full", "rgb_only", "preserve_alpha")
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
                
                # 应用颜色反向操作
                reversed_img = self._apply_color_reverse(img_np, reverse_mode)
                numpy_images.append(reversed_img)
            
            # 转换回torch张量
            output_tensor = torch.from_numpy(np.array(numpy_images)).float()
            
            # 输出使用说明
            if usage_notes:
                print(f"=== TT Image Color Reverse 使用说明 ===")
                print(usage_notes)
                print(f"=== 处理完成 ===")
                print(f"反转模式: {reverse_mode}")
                print(f"处理图像数量: {len(images)}")
                print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in TT Image Color Reverse node: {str(e)}")
            # 返回原始图像作为错误处理
            return (images,)
    
    def _apply_color_reverse(self, image: np.ndarray, reverse_mode: str) -> np.ndarray:
        """
        对单张图像应用颜色反向操作
        
        Args:
            image: 输入图像数组 (height, width, channels)
            reverse_mode: 反转模式
        
        Returns:
            np.ndarray: 处理后的图像数组
        """
        try:
            # 复制图像以避免修改原始数据
            result = image.copy()
            
            if reverse_mode == "full":
                # 完全反转：所有通道都进行反转
                result = 1.0 - result
                
            elif reverse_mode == "rgb_only":
                # 仅RGB反转：只反转RGB通道，保留透明度通道
                if result.shape[2] >= 3:
                    # 反转RGB通道
                    result[:, :, :3] = 1.0 - result[:, :, :3]
                    # 如果有alpha通道，保持不变
                    if result.shape[2] == 4:
                        result[:, :, 3] = image[:, :, 3]  # 保持原始alpha值
                else:
                    # 如果只有1-2个通道，全部反转
                    result = 1.0 - result
                    
            elif reverse_mode == "preserve_alpha":
                # 保留透明度反转：反转RGB，但保持alpha通道不变
                if result.shape[2] == 4:
                    # 有alpha通道的情况
                    result[:, :, :3] = 1.0 - result[:, :, :3]
                    result[:, :, 3] = image[:, :, 3]  # 保持原始alpha值
                elif result.shape[2] == 3:
                    # 只有RGB通道，全部反转
                    result = 1.0 - result
                else:
                    # 其他情况，全部反转
                    result = 1.0 - result
            else:
                # 默认情况，完全反转
                result = 1.0 - result
            
            # 确保值在有效范围内
            result = np.clip(result, 0.0, 1.0)
            
            return result
                
        except Exception as e:
            print(f"Error in _apply_color_reverse: {str(e)}")
            return image

# 节点类定义完成
