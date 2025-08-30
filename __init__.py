"""
TT img enc - ComfyUI 自定义节点
自动图片格式转换和文件隐写术
"""

from .tt_img_enc_node import TTImgEncNode

# 注册节点类
NODE_CLASS_MAPPINGS = {
    "TT_img_enc": TTImgEncNode
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TT_img_enc": "TT img enc"
}

# 节点分类
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
