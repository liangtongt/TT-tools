"""
ComfyUI Image Sequence Compression Node
图片序列压缩节点 - 将多张图片压缩到一个文件中
"""

from .image_sequence_compressor import ImageSequenceCompressor

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageSequenceCompressor": ImageSequenceCompressor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceCompressor": "Image Sequence Compressor"
}

# 节点描述
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"

print(f"Image Sequence Compressor Node v{__version__} 已加载")
