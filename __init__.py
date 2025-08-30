"""
ComfyUI TT img Node
TT img 节点 - 将图片序列压缩到一个文件中
"""

from .image_sequence_compressor import TTImg

# 注册节点
NODE_CLASS_MAPPINGS = {
    "TTImg": TTImg
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TTImg": "TT img"
}

# 节点描述
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"

print(f"TT img Node v{__version__} 已加载")
